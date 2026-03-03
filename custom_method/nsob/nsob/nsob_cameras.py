from nerfstudio.cameras.cameras import *
from nsob.nsob_rays import NsobRayBundle

class NsobCameras(Cameras):
    """Dataparser outputs for the image dataset and the ray generator.
    """
    def __init__(
        self,
        camera_to_worlds: Float[Tensor, "*batch_c2ws 3 4"],
        fx: Union[Float[Tensor, "*batch_fxs 1"], float],
        fy: Union[Float[Tensor, "*batch_fys 1"], float],
        cx: Union[Float[Tensor, "*batch_cxs 1"], float],
        cy: Union[Float[Tensor, "*batch_cys 1"], float],
        width: Optional[Union[Shaped[Tensor, "*batch_ws 1"], int]] = None,
        height: Optional[Union[Shaped[Tensor, "*batch_hs 1"], int]] = None,
        distortion_params: Optional[Float[Tensor, "*batch_dist_params 6"]] = None,
        camera_type: Union[
            Int[Tensor, "*batch_cam_types 1"],
            int,
            List[CameraType],
            CameraType,
        ] = CameraType.PERSPECTIVE,
        times: Optional[Float[Tensor, "num_cameras"]] = None,
        metadata: Optional[Dict] = None,
    ) -> None:
        super().__init__(
            camera_to_worlds = camera_to_worlds,
            fx = fx,
            fy = fy,
            cx = cx,
            cy = cy,
            width = width,
            height = height,
            distortion_params = distortion_params,
            camera_type = camera_type,
            times = times,
            metadata = metadata,
        )

    def _generate_rays_from_coords(
        self,
        camera_indices: Int[Tensor, "*num_rays num_cameras_batch_dims"],
        coords: Float[Tensor, "*num_rays 2"],
        camera_opt_to_camera: Optional[Float[Tensor, "*num_rays 3 4"]] = None,
        distortion_params_delta: Optional[Float[Tensor, "*num_rays 6"]] = None,
        disable_distortion: bool = False,
    ) -> RayBundle:
        """Generates rays for the given camera indices and coords where self isn't jagged

        Returns:
            Rays for the given camera indices and coords. RayBundle.shape == num_rays
        """
        # Make sure we're on the right devices
        camera_indices = camera_indices.to(self.device)
        coords = coords.to(self.device)

        # Checking to make sure everything is of the right shape and type
        num_rays_shape = camera_indices.shape[:-1]
        assert camera_indices.shape == num_rays_shape + (self.ndim,)
        assert coords.shape == num_rays_shape + (2,)
        assert coords.shape[-1] == 2
        assert camera_opt_to_camera is None or camera_opt_to_camera.shape == num_rays_shape + (3, 4)
        assert distortion_params_delta is None or distortion_params_delta.shape == num_rays_shape + (6,)

        # Here, we've broken our indices down along the num_cameras_batch_dims dimension allowing us to index by all
        # of our output rays at each dimension of our cameras object
        true_indices = [camera_indices[..., i] for i in range(camera_indices.shape[-1])]

        # Get all our focal lengths, principal points and make sure they are the right shapes
        y = coords[..., 0]  # (num_rays,) get rid of the last dimension
        x = coords[..., 1]  # (num_rays,) get rid of the last dimension
        fx, fy = self.fx[true_indices].squeeze(-1), self.fy[true_indices].squeeze(-1)  # (num_rays,)
        cx, cy = self.cx[true_indices].squeeze(-1), self.cy[true_indices].squeeze(-1)  # (num_rays,)
        assert (
            y.shape == num_rays_shape
            and x.shape == num_rays_shape
            and fx.shape == num_rays_shape
            and fy.shape == num_rays_shape
            and cx.shape == num_rays_shape
            and cy.shape == num_rays_shape
        ), (
            str(num_rays_shape)
            + str(y.shape)
            + str(x.shape)
            + str(fx.shape)
            + str(fy.shape)
            + str(cx.shape)
            + str(cy.shape)
        )

        # Get our image coordinates and image coordinates offset by 1 (offsets used for dx, dy calculations)
        # Also make sure the shapes are correct
        coord = torch.stack([(x - cx) / fx, -(y - cy) / fy], -1)  # (num_rays, 2)
        coord_x_offset = torch.stack([(x - cx + 1) / fx, -(y - cy) / fy], -1)  # (num_rays, 2)
        coord_y_offset = torch.stack([(x - cx) / fx, -(y - cy + 1) / fy], -1)  # (num_rays, 2)
        assert (
            coord.shape == num_rays_shape + (2,)
            and coord_x_offset.shape == num_rays_shape + (2,)
            and coord_y_offset.shape == num_rays_shape + (2,)
        )

        # Stack image coordinates and image coordinates offset by 1, check shapes too
        coord_stack = torch.stack([coord, coord_x_offset, coord_y_offset], dim=0)  # (3, num_rays, 2)
        assert coord_stack.shape == (3,) + num_rays_shape + (2,)

        # Undistorts our images according to our distortion parameters
        if not disable_distortion:
            distortion_params = None
            if self.distortion_params is not None:
                distortion_params = self.distortion_params[true_indices]
                if distortion_params_delta is not None:
                    distortion_params = distortion_params + distortion_params_delta
            elif distortion_params_delta is not None:
                distortion_params = distortion_params_delta

            # Do not apply distortion for equirectangular images
            if distortion_params is not None:
                mask = (self.camera_type[true_indices] != CameraType.EQUIRECTANGULAR.value).squeeze(-1)  # (num_rays)
                coord_mask = torch.stack([mask, mask, mask], dim=0)
                if mask.any() and (distortion_params != 0).any():
                    coord_stack[coord_mask, :] = camera_utils.radial_and_tangential_undistort(
                        coord_stack[coord_mask, :].reshape(3, -1, 2),
                        distortion_params[mask, :],
                    ).reshape(-1, 2)

        # Make sure after we have undistorted our images, the shapes are still correct
        assert coord_stack.shape == (3,) + num_rays_shape + (2,)

        # Gets our directions for all our rays in camera coordinates and checks shapes at the end
        # Here, directions_stack is of shape (3, num_rays, 3)
        # directions_stack[0] is the direction for ray in camera coordinates
        # directions_stack[1] is the direction for ray in camera coordinates offset by 1 in x
        # directions_stack[2] is the direction for ray in camera coordinates offset by 1 in y
        cam_types = torch.unique(self.camera_type, sorted=False)
        directions_stack = torch.empty((3,) + num_rays_shape + (3,), device=self.device)

        c2w = self.camera_to_worlds[true_indices]
        assert c2w.shape == num_rays_shape + (3, 4)

        def _compute_rays_for_omnidirectional_stereo(
            eye: Literal["left", "right"]
        ) -> Tuple[Float[Tensor, "num_rays_shape 3"], Float[Tensor, "3 num_rays_shape 3"]]:
            """Compute the rays for an omnidirectional stereo camera

            Args:
                eye: Which eye to compute rays for.

            Returns:
                A tuple containing the origins and the directions of the rays.
            """
            # Directions calculated similarly to equirectangular
            ods_cam_type = (
                CameraType.OMNIDIRECTIONALSTEREO_R.value if eye == "right" else CameraType.OMNIDIRECTIONALSTEREO_L.value
            )
            mask = (self.camera_type[true_indices] == ods_cam_type).squeeze(-1)
            mask = torch.stack([mask, mask, mask], dim=0)
            theta = -torch.pi * coord_stack[..., 0]
            phi = torch.pi * (0.5 - coord_stack[..., 1])

            directions_stack[..., 0][mask] = torch.masked_select(-torch.sin(theta) * torch.sin(phi), mask).float()
            directions_stack[..., 1][mask] = torch.masked_select(torch.cos(phi), mask).float()
            directions_stack[..., 2][mask] = torch.masked_select(-torch.cos(theta) * torch.sin(phi), mask).float()

            vr_ipd = 0.064  # IPD in meters (note: scale of NeRF must be true to life and can be adjusted with the Blender add-on)
            isRightEye = 1 if eye == "right" else -1

            # find ODS camera position
            c2w = self.camera_to_worlds[true_indices]
            assert c2w.shape == num_rays_shape + (3, 4)
            transposedC2W = c2w[0][0].t()
            ods_cam_position = transposedC2W[3].repeat(c2w.shape[1], 1)

            rotation = c2w[..., :3, :3]

            ods_theta = -torch.pi * ((x - cx) / fx)[0]

            # local axes of ODS camera
            ods_x_axis = torch.tensor([1, 0, 0], device=c2w.device)
            ods_z_axis = torch.tensor([0, 0, -1], device=c2w.device)

            # circle of ODS ray origins
            ods_origins_circle = (
                isRightEye * (vr_ipd / 2.0) * (ods_x_axis.repeat(c2w.shape[1], 1)) * (torch.cos(ods_theta))[:, None]
                + isRightEye * (vr_ipd / 2.0) * (ods_z_axis.repeat(c2w.shape[1], 1)) * (torch.sin(ods_theta))[:, None]
            )

            # rotate origins to match the camera rotation
            for i in range(ods_origins_circle.shape[0]):
                ods_origins_circle[i] = rotation[0][0] @ ods_origins_circle[i] + ods_cam_position[0]
            ods_origins_circle = ods_origins_circle.unsqueeze(0).repeat(c2w.shape[0], 1, 1)

            # assign final camera origins
            c2w[..., :3, 3] = ods_origins_circle

            return ods_origins_circle, directions_stack

        for cam in cam_types:
            if CameraType.PERSPECTIVE.value in cam_types:
                mask = (self.camera_type[true_indices] == CameraType.PERSPECTIVE.value).squeeze(-1)  # (num_rays)
                mask = torch.stack([mask, mask, mask], dim=0)
                directions_stack[..., 0][mask] = torch.masked_select(coord_stack[..., 0], mask).float()
                directions_stack[..., 1][mask] = torch.masked_select(coord_stack[..., 1], mask).float()
                directions_stack[..., 2][mask] = -1.0

            elif CameraType.FISHEYE.value in cam_types:
                mask = (self.camera_type[true_indices] == CameraType.FISHEYE.value).squeeze(-1)  # (num_rays)
                mask = torch.stack([mask, mask, mask], dim=0)

                theta = torch.sqrt(torch.sum(coord_stack**2, dim=-1))
                theta = torch.clip(theta, 0.0, math.pi)

                sin_theta = torch.sin(theta)

                directions_stack[..., 0][mask] = torch.masked_select(
                    coord_stack[..., 0] * sin_theta / theta, mask
                ).float()
                directions_stack[..., 1][mask] = torch.masked_select(
                    coord_stack[..., 1] * sin_theta / theta, mask
                ).float()
                directions_stack[..., 2][mask] = -torch.masked_select(torch.cos(theta), mask).float()

            elif CameraType.EQUIRECTANGULAR.value in cam_types:
                mask = (self.camera_type[true_indices] == CameraType.EQUIRECTANGULAR.value).squeeze(-1)  # (num_rays)
                mask = torch.stack([mask, mask, mask], dim=0)

                # For equirect, fx = fy = height = width/2
                # Then coord[..., 0] goes from -1 to 1 and coord[..., 1] goes from -1/2 to 1/2
                theta = -torch.pi * coord_stack[..., 0]  # minus sign for right-handed
                phi = torch.pi * (0.5 - coord_stack[..., 1])
                # use spherical in local camera coordinates (+y up, x=0 and z<0 is theta=0)
                directions_stack[..., 0][mask] = torch.masked_select(-torch.sin(theta) * torch.sin(phi), mask).float()
                directions_stack[..., 1][mask] = torch.masked_select(torch.cos(phi), mask).float()
                directions_stack[..., 2][mask] = torch.masked_select(-torch.cos(theta) * torch.sin(phi), mask).float()

            elif CameraType.OMNIDIRECTIONALSTEREO_L.value in cam_types:
                ods_origins_circle, directions_stack = _compute_rays_for_omnidirectional_stereo("left")
                # assign final camera origins
                c2w[..., :3, 3] = ods_origins_circle

            elif CameraType.OMNIDIRECTIONALSTEREO_R.value in cam_types:
                ods_origins_circle, directions_stack = _compute_rays_for_omnidirectional_stereo("right")
                # assign final camera origins
                c2w[..., :3, 3] = ods_origins_circle
            else:
                raise ValueError(f"Camera type {cam} not supported.")

        assert directions_stack.shape == (3,) + num_rays_shape + (3,)

        if camera_opt_to_camera is not None:
            c2w = pose_utils.multiply(c2w, camera_opt_to_camera)
        rotation = c2w[..., :3, :3]  # (..., 3, 3)
        assert rotation.shape == num_rays_shape + (3, 3)

        directions_stack = torch.sum(
            directions_stack[..., None, :] * rotation, dim=-1
        )  # (..., 1, 3) * (..., 3, 3) -> (..., 3)
        directions_stack, directions_norm = camera_utils.normalize_with_norm(directions_stack, -1)
        assert directions_stack.shape == (3,) + num_rays_shape + (3,)

        origins = c2w[..., :3, 3]  # (..., 3)
        assert origins.shape == num_rays_shape + (3,)

        directions = directions_stack[0]
        assert directions.shape == num_rays_shape + (3,)

        # norms of the vector going between adjacent coords, giving us dx and dy per output ray
        dx = torch.sqrt(torch.sum((directions - directions_stack[1]) ** 2, dim=-1))  # ("num_rays":...,)
        dy = torch.sqrt(torch.sum((directions - directions_stack[2]) ** 2, dim=-1))  # ("num_rays":...,)
        assert dx.shape == num_rays_shape and dy.shape == num_rays_shape

        pixel_area = (dx * dy)[..., None]  # ("num_rays":..., 1)
        assert pixel_area.shape == num_rays_shape + (1,)

        times = self.times[camera_indices, 0] if self.times is not None else None

        metadata = (
            self._apply_fn_to_dict(self.metadata, lambda x: x[true_indices]) if self.metadata is not None else None
        )
        if metadata is not None:
            metadata["directions_norm"] = directions_norm[0].detach()
        else:
            metadata = {"directions_norm": directions_norm[0].detach()}

        times = self.times[camera_indices, 0] if self.times is not None else None

        return NsobRayBundle(
            origins=origins,
            directions=directions,
            pixel_area=pixel_area,
            camera_indices=camera_indices,
            times=times,
            metadata=metadata,
        )