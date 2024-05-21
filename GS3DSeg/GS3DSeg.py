
from dataclasses import dataclass, field
from nerfstudio.cameras.cameras import Cameras
import torch
from torch.nn import Parameter
from nerfstudio.models.splatfacto import SplatfactoModel, SplatfactoModelConfig 
import torch.nn.functional as F
from gsplat.rasterize import rasterize_gaussians
from gsplat._torch_impl import quat_to_rotmat
from nerfstudio.engine.optimizers import Optimizers
import math
from dataclasses import dataclass, field
from typing import Dict, List, Type, Union, Optional
from gsplat.sh import num_sh_bases, spherical_harmonics
import numpy as np
import torch
from gsplat.project_gaussians import project_gaussians
from gsplat.rasterize import rasterize_gaussians
from gsplat.sh import spherical_harmonics
from tqdm import tqdm
from nerfstudio.cameras.camera_optimizers import CameraOptimizer, CameraOptimizerConfig
import torch.bin
DEVICE = 'cuda'
# need following import for background color override
from nerfstudio.model_components import renderers
from nerfstudio.utils.rich_utils import CONSOLE
def projection_matrix(znear, zfar, fovx, fovy, device: Union[str, torch.device] = "cpu"):
    """
    Constructs an OpenGL-style perspective projection matrix.
    """
    t = znear * math.tan(0.5 * fovy)
    b = -t
    r = znear * math.tan(0.5 * fovx)
    l = -r
    n = znear
    f = zfar
    return torch.tensor(
        [
            [2 * n / (r - l), 0.0, (r + l) / (r - l), 0.0],
            [0.0, 2 * n / (t - b), (t + b) / (t - b), 0.0],
            [0.0, 0.0, (f + n) / (f - n), -1.0 * f * n / (f - n)],
            [0.0, 0.0, 1.0, 0.0],
        ],
        device=device,
    )

@dataclass
class GS3DSegConfig(SplatfactoModelConfig):
    _target: Type = field(default_factory=lambda: GS3DSeg)
    camera_optimizer: CameraOptimizerConfig = field(default_factory=lambda: CameraOptimizerConfig(mode="off"))
    num_random = 50000



class GS3DSeg(SplatfactoModel):

    config: GS3DSegConfig

    def populate_modules(self):
        super().populate_modules()
        self.epoch = 1
        self.identity_vec = torch.nn.Parameter(torch.rand([self.num_points, 16]))
         # enter path to Sam embeddings
        self.embd_path = '/scratch/ashwin/gsplat/scene1/Sam_annotations/final.npy'
        self.identity = torch.from_numpy(np.load(self.embd_path)).to(DEVICE).type(torch.uint8)
        num_images = self.identity.shape[0]
        self.identity = self.identity.reshape(num_images, -1)
        self.camera_optimizer: CameraOptimizer = self.config.camera_optimizer.setup(
            num_cameras=self.num_train_data, device="cpu"
        )
        
        # self.train_image_list  = [p.name for p in self.train_image_list]
        # torch_embd = torch.ones([1,738, 994 ])
        # for im in tqdm(self.train_image_list):
        #     embd = torch.from_numpy(np.load(embd_path+'/'+im+'.npy'))
        #     torch_embd = torch.cat([torch_embd, embd.unsqueeze(0)], dim=0)
        # self.torch_embd = torch_embd[1:, :, :].type(dtype=torch.uint8)
    
    # def load_state_dict(self, dict, **kwargs):
    #     newp = dict["means"].shape[0]
    #     self.identity_vec = torch.nn.Parameter(torch.rand(newp, 16, device=self.device))
    #     super().load_state_dict(dict, **kwargs)

    def load_state_dict(self, dict, **kwargs):  # type: ignore
        # resize the parameters to match the new number of points
        self.step = 30000
        newp = dict["means"].shape[0]
        self.means = torch.nn.Parameter(torch.zeros(newp, 3, device=self.device))
        self.scales = torch.nn.Parameter(torch.zeros(newp, 3, device=self.device))
        self.quats = torch.nn.Parameter(torch.zeros(newp, 4, device=self.device))
        self.opacities = torch.nn.Parameter(torch.zeros(newp, 1, device=self.device))
        self.features_dc = torch.nn.Parameter(torch.zeros(newp, 3, device=self.device))
        self.features_rest = torch.nn.Parameter(
            torch.zeros(newp, num_sh_bases(self.config.sh_degree) - 1, 3, device=self.device)
        )
        self.identity_vec = torch.nn.Parameter(torch.rand(newp, 16, device=self.device))
        super().load_state_dict(dict, **kwargs)
    
    def get_gaussian_param_groups(self) -> Dict[str, List[Parameter]]:
        return {
            "xyz": [self.means],
            "features_dc": [self.features_dc],
            "features_rest": [self.features_rest],
            "opacity": [self.opacities],
            "scaling": [self.scales],
            "rotation": [self.quats],
            "identity_vec": [self.identity_vec]
        }
    
    def get_outputs(self, camera: Cameras) -> Dict[str, Union[torch.Tensor, List]]:
        """Takes in a Ray Bundle and returns a dictionary of outputs.

        Args:
            ray_bundle: Input bundle of rays. This raybundle should have all the
            needed information to compute the outputs.

        Returns:
            Outputs of model. (ie. rendered colors)
        """
        if not isinstance(camera, Cameras):
            print("Called get_outputs with not a camera")
            return {}
        assert camera.shape[0] == 1, "Only one camera at a time"

        # get the background color
        if self.training:
            if self.config.background_color == "random":
                background = torch.rand(3, device=self.device)
            elif self.config.background_color == "white":
                background = torch.ones(3, device=self.device)
            elif self.config.background_color == "black":
                background = torch.zeros(3, device=self.device)
            else:
                background = self.background_color.to(self.device)
        else:
            if renderers.BACKGROUND_COLOR_OVERRIDE is not None:
                background = renderers.BACKGROUND_COLOR_OVERRIDE.to(self.device)
            else:
                background = self.background_color.to(self.device)

        if self.crop_box is not None and not self.training:
            crop_ids = self.crop_box.within(self.means).squeeze()
            if crop_ids.sum() == 0:
                return {"rgb": background.repeat(int(camera.height.item()), int(camera.width.item()), 1)}
        else:
            crop_ids = None
        camera_downscale = self._get_downscale_factor()
        camera.rescale_output_resolution(1 / camera_downscale)
        # shift the camera to center of scene looking at center
        R = camera.camera_to_worlds[0, :3, :3]  # 3 x 3
        T = camera.camera_to_worlds[0, :3, 3:4]  # 3 x 1
        # flip the z and y axes to align with gsplat conventions
        R_edit = torch.diag(torch.tensor([1, -1, -1], device=self.device, dtype=R.dtype))
        R = R @ R_edit
        # analytic matrix inverse to get world2camera matrix
        R_inv = R.T
        T_inv = -R_inv @ T
        viewmat = torch.eye(4, device=R.device, dtype=R.dtype)
        viewmat[:3, :3] = R_inv
        viewmat[:3, 3:4] = T_inv
        # calculate the FOV of the camera given fx and fy, width and height
        cx = camera.cx.item()
        cy = camera.cy.item()
        fovx = 2 * math.atan(camera.width / (2 * camera.fx))
        fovy = 2 * math.atan(camera.height / (2 * camera.fy))
        W, H = int(camera.width.item()), int(camera.height.item())
        self.last_size = (H, W)
        projmat = projection_matrix(0.001, 1000, fovx, fovy, device=self.device)
        BLOCK_X, BLOCK_Y = 16, 16
        tile_bounds = (
            int((W + BLOCK_X - 1) // BLOCK_X),
            int((H + BLOCK_Y - 1) // BLOCK_Y),
            1,
        )

        if crop_ids is not None:
            opacities_crop = self.opacities[crop_ids]
            means_crop = self.means[crop_ids]
            features_dc_crop = self.features_dc[crop_ids]
            features_rest_crop = self.features_rest[crop_ids]
            scales_crop = self.scales[crop_ids]
            quats_crop = self.quats[crop_ids]
        else:
            opacities_crop = self.opacities
            means_crop = self.means
            features_dc_crop = self.features_dc
            features_rest_crop = self.features_rest
            scales_crop = self.scales
            quats_crop = self.quats

        colors_crop = torch.cat((features_dc_crop[:, None, :], features_rest_crop), dim=1)

        self.xys, depths, self.radii, conics, num_tiles_hit, cov3d = project_gaussians(  # type: ignore
            means_crop,
            torch.exp(scales_crop),
            1,
            quats_crop / quats_crop.norm(dim=-1, keepdim=True),
            viewmat.squeeze()[:3, :],
            projmat.squeeze() @ viewmat.squeeze(),
            camera.fx.item(),
            camera.fy.item(),
            cx,
            cy,
            H,
            W,
            tile_bounds,
        )  # type: ignore
        if (self.radii).sum() == 0:
            return {"rgb": background.repeat(int(camera.height.item()), int(camera.width.item()), 1)}

        # Important to allow xys grads to populate properly
        if self.training:
            self.xys.retain_grad()

        if self.config.sh_degree > 0:
            viewdirs = means_crop.detach() - camera.camera_to_worlds.detach()[..., :3, 3]  # (N, 3)
            viewdirs = viewdirs / viewdirs.norm(dim=-1, keepdim=True)
            n = min(self.step // self.config.sh_degree_interval, self.config.sh_degree)
            rgbs = spherical_harmonics(n, viewdirs, colors_crop)
            rgbs = torch.clamp(rgbs + 0.5, min=0.0)  # type: ignore
        else:
            rgbs = torch.sigmoid(colors_crop[:, 0, :])

        # rescale the camera back to original dimensions
        camera.rescale_output_resolution(camera_downscale)
        assert (num_tiles_hit > 0).any()  # type: ignore
        rgb = rasterize_gaussians(  # type: ignore
            self.xys,
            depths,
            self.radii,
            conics,
            num_tiles_hit,  # type: ignore
            rgbs,
            torch.sigmoid(opacities_crop),
            H,
            W,
            background=background,
        )  # type: ignore
        rgb = torch.clamp(rgb, max=1.0)  # type: ignore
        depth_im = None
        if not self.training:
            depth_im = rasterize_gaussians(  # type: ignore
                self.xys,
                depths,
                self.radii,
                conics,
                num_tiles_hit,  # type: ignore
                depths[:, None].repeat(1, 3),
                torch.sigmoid(opacities_crop),
                H,
                W,
                background=torch.ones(3, device=self.device) * 10,
            )[..., 0:1]  # type: ignore
        data = {"rgb": rgb, "depth": depth_im} 
        identity_norm = F.normalize(self.identity_vec, dim = -1)
        W, H = int(camera.width.item()), int(camera.height.item())
        identity = rasterize_gaussians(  # type: ignore
            self.xys,
            depths,
            self.radii,
            conics,
            num_tiles_hit,  # type: ignore
            identity_norm,
            torch.sigmoid(opacities_crop),
            H,
            W,
            background=torch.ones(16, device=self.device) * 10,
        )
        data['identity'] = identity
        return data # type: ignore
    
    def split_gaussians(self, split_mask, samps):
        """
        This function splits gaussians that are too large
        """

        n_splits = split_mask.sum().item()
        CONSOLE.log(f"Splitting {split_mask.sum().item()/self.num_points} gaussians: {n_splits}/{self.num_points}")
        centered_samples = torch.randn((samps * n_splits, 3), device=self.device)  # Nx3 of axis-aligned scales
        scaled_samples = (
            torch.exp(self.scales[split_mask].repeat(samps, 1)) * centered_samples
        )  # how these scales are rotated
        quats = self.quats[split_mask] / self.quats[split_mask].norm(dim=-1, keepdim=True)  # normalize them first
        rots = quat_to_rotmat(quats.repeat(samps, 1))  # how these scales are rotated
        rotated_samples = torch.bmm(rots, scaled_samples[..., None]).squeeze()
        new_means = rotated_samples + self.means[split_mask].repeat(samps, 1)
        # step 2, sample new colors
        new_features_dc = self.features_dc[split_mask].repeat(samps, 1)
        new_features_rest = self.features_rest[split_mask].repeat(samps, 1, 1)
        # step 3, sample new opacities
        new_opacities = self.opacities[split_mask].repeat(samps, 1)
        # step 4, sample new scales
        size_fac = 1.6
        new_scales = torch.log(torch.exp(self.scales[split_mask]) / size_fac).repeat(samps, 1)
        self.scales[split_mask] = torch.log(torch.exp(self.scales[split_mask]) / size_fac)
        # step 5, sample new quats
        new_quats = self.quats[split_mask].repeat(samps, 1)
        # step 6 Sample new identities
        new_identity = self.identity_vec[split_mask].repeat(samps, 1)
        return (
            new_means,
            new_features_dc,
            new_features_rest,
            new_opacities,
            new_scales,
            new_quats,
            new_identity,
        )

    def dup_gaussians(self, dup_mask):
        """
        This function duplicates gaussians that are too small
        """
        n_dups = dup_mask.sum().item()
        CONSOLE.log(f"Duplicating {dup_mask.sum().item()/self.num_points} gaussians: {n_dups}/{self.num_points}")
        dup_means = self.means[dup_mask]
        dup_features_dc = self.features_dc[dup_mask]
        dup_features_rest = self.features_rest[dup_mask]
        dup_opacities = self.opacities[dup_mask]
        dup_scales = self.scales[dup_mask]
        dup_quats = self.quats[dup_mask]
        dup_identity = self.identity_vec[dup_mask]
        return (
            dup_means,
            dup_features_dc,
            dup_features_rest,
            dup_opacities,
            dup_scales,
            dup_quats,
            dup_identity,
        )
    
    def refinement_after(self, optimizers: Optimizers, step):
        assert step == self.step
        if self.step <= self.config.warmup_length:
            return
        with torch.no_grad():
            # Offset all the opacity reset logic by refine_every so that we don't
            # save checkpoints right when the opacity is reset (saves every 2k)
            # then cull
            # only split/cull if we've seen every image since opacity reset
            reset_interval = self.config.reset_alpha_every * self.config.refine_every
            do_densification = (
                self.step < self.config.stop_split_at
                and self.step % reset_interval > self.num_train_data + self.config.refine_every
            )
            if do_densification:
                # then we densify
                assert self.xys_grad_norm is not None and self.vis_counts is not None and self.max_2Dsize is not None
                avg_grad_norm = (self.xys_grad_norm / self.vis_counts) * 0.5 * max(self.last_size[0], self.last_size[1])
                high_grads = (avg_grad_norm > self.config.densify_grad_thresh).squeeze()
                splits = (self.scales.exp().max(dim=-1).values > self.config.densify_size_thresh).squeeze()
                if self.step < self.config.stop_screen_size_at:
                    splits |= (self.max_2Dsize > self.config.split_screen_size).squeeze()
                splits &= high_grads
                nsamps = self.config.n_split_samples
                (
                    split_means,
                    split_features_dc,
                    split_features_rest,
                    split_opacities,
                    split_scales,
                    split_quats,
                    split_identity,
                ) = self.split_gaussians(splits, nsamps)

                dups = (self.scales.exp().max(dim=-1).values <= self.config.densify_size_thresh).squeeze()
                dups &= high_grads
                (
                    dup_means,
                    dup_features_dc,
                    dup_features_rest,
                    dup_opacities,
                    dup_scales,
                    dup_quats,
                    dup_identity,
                ) = self.dup_gaussians(dups)
                self.means = Parameter(torch.cat([self.means.detach(), split_means, dup_means], dim=0))
                self.features_dc = Parameter(
                    torch.cat(
                        [self.features_dc.detach(), split_features_dc, dup_features_dc],
                        dim=0,
                    )
                )
                self.features_rest = Parameter(
                    torch.cat(
                        [
                            self.features_rest.detach(),
                            split_features_rest,
                            dup_features_rest,
                        ],
                        dim=0,
                    )
                )
                self.opacities = Parameter(torch.cat([self.opacities.detach(), split_opacities, dup_opacities], dim=0))
                self.scales = Parameter(torch.cat([self.scales.detach(), split_scales, dup_scales], dim=0))
                self.quats = Parameter(torch.cat([self.quats.detach(), split_quats, dup_quats], dim=0))
                self.identity_vec = Parameter(torch.cat([self.identity_vec.detach(), split_identity, dup_identity], dim = 0))
                # append zeros to the max_2Dsize tensor
                self.max_2Dsize = torch.cat(
                    [
                        self.max_2Dsize,
                        torch.zeros_like(split_scales[:, 0]),
                        torch.zeros_like(dup_scales[:, 0]),
                    ],
                    dim=0,
                )

                split_idcs = torch.where(splits)[0]
                self.dup_in_all_optim(optimizers, split_idcs, nsamps)

                dup_idcs = torch.where(dups)[0]
                self.dup_in_all_optim(optimizers, dup_idcs, 1)

                # After a guassian is split into two new gaussians, the original one should also be pruned.
                splits_mask = torch.cat(
                    (
                        splits,
                        torch.zeros(
                            nsamps * splits.sum() + dups.sum(),
                            device=self.device,
                            dtype=torch.bool,
                        ),
                    )
                )

                deleted_mask = self.cull_gaussians(splits_mask)
            elif self.step >= self.config.stop_split_at and self.config.continue_cull_post_densification:
                deleted_mask = self.cull_gaussians()
            else:
                # if we donot allow culling post refinement, no more gaussians will be pruned.
                deleted_mask = None

            if deleted_mask is not None:
                self.remove_from_all_optim(optimizers, deleted_mask)

            if self.step < self.config.stop_split_at and self.step % reset_interval == self.config.refine_every:
                # Reset value is set to be twice of the cull_alpha_thresh
                reset_value = self.config.cull_alpha_thresh * 2.0
                self.opacities.data = torch.clamp(
                    self.opacities.data,
                    max=torch.logit(torch.tensor(reset_value, device=self.device)).item(),
                )
                # reset the exp of optimizer
                optim = optimizers.optimizers["opacity"]
                param = optim.param_groups[0]["params"][0]
                param_state = optim.state[param]
                param_state["exp_avg"] = torch.zeros_like(param_state["exp_avg"])
                param_state["exp_avg_sq"] = torch.zeros_like(param_state["exp_avg_sq"])

            self.xys_grad_norm = None
            self.vis_counts = None
            self.max_2Dsize = None

    def cull_gaussians(self, extra_cull_mask: Optional[torch.Tensor] = None):
        """
        This function deletes gaussians with under a certain opacity threshold
        extra_cull_mask: a mask indicates extra gaussians to cull besides existing culling criterion
        """
        n_bef = self.num_points
        # cull transparent ones
        culls = (torch.sigmoid(self.opacities) < self.config.cull_alpha_thresh).squeeze()
        below_alpha_count = torch.sum(culls).item()
        toobigs_count = 0
        if extra_cull_mask is not None:
            culls = culls | extra_cull_mask
        if self.step > self.config.refine_every * self.config.reset_alpha_every:
            # cull huge ones
            toobigs = (torch.exp(self.scales).max(dim=-1).values > self.config.cull_scale_thresh).squeeze()
            if self.step < self.config.stop_screen_size_at:
                # cull big screen space
                assert self.max_2Dsize is not None
                toobigs = toobigs | (self.max_2Dsize > self.config.cull_screen_size).squeeze()
            culls = culls | toobigs
            toobigs_count = torch.sum(toobigs).item()
        self.means = Parameter(self.means[~culls].detach())
        self.scales = Parameter(self.scales[~culls].detach())
        self.quats = Parameter(self.quats[~culls].detach())
        self.features_dc = Parameter(self.features_dc[~culls].detach())
        self.features_rest = Parameter(self.features_rest[~culls].detach())
        self.opacities = Parameter(self.opacities[~culls].detach())
        self.identity_vec=Parameter(self.identity_vec[~culls].detach())

        CONSOLE.log(
            f"Culled {n_bef - self.num_points} gaussians "
            f"({below_alpha_count} below alpha thresh, {toobigs_count} too bigs, {self.num_points} remaining)"
        )
        return culls
    
    def identity_loss(self, outputs, batch):
        predicted_identity = outputs['identity']
        gt_identity = self.identity[batch['image_idx'], :, :]
        assert predicted_identity.shape[:2] == gt_identity.shape[:2]
        num_masks = torch.max(gt_identity)
        Pos, Neg = torch.tensor([0.], device=DEVICE), torch.tensor([0.], device=DEVICE)
        for i in range(num_masks):
            mask = gt_identity.view(-1) == i
            _pos_matrix = predicted_identity.view(-1, 16)[mask]
            _pos_matrix = _pos_matrix[torch.randperm(_pos_matrix.shape[0])[:100]]
            _pos_matrix = torch.nn.functional.normalize(_pos_matrix, dim=1)
            pos_matrix = torch.mm(_pos_matrix, _pos_matrix.T)
            # assert torch.all(pos_matrix <= 1.5)
            Pos += torch.mean(1 - pos_matrix)
            neg_matrix = predicted_identity.view(-1, 16)[~mask]
            neg_matrix = neg_matrix[torch.randperm(neg_matrix.shape[0])[:200]]
            neg_matrix = torch.nn.functional.normalize(neg_matrix, dim=1)
            neg_matrix = torch.mm(neg_matrix, _pos_matrix.T)
            # assert torch.all(neg_matrix <= 1.5)
            neg_matrix = torch.nn.ReLU()(neg_matrix - 0.5)
            Neg += torch.mean(neg_matrix)
        return Pos, Neg
    
    def identity_loss_optimised(self, pred_identity, image_id):
        # pred_identity ---> HW x 16
        pred_identity = pred_identity.view(-1, 16)
        #randomly extracting freatures index
        rand_ind = torch.randperm(pred_identity.shape[0])[:10_000]
        pred_identity = pred_identity[rand_ind]
        identity = self.identity[image_id][rand_ind]
        sorted_index, indices = torch.sort(identity)
        num_masks = torch.bincount(sorted_index)
        cumsum_index = torch.cumsum(num_masks, dim=0)
        pred_identity_ordered = torch.gather(pred_identity, dim=0, index=indices.view(-1, 1).expand(-1, 16))
        identity_vec_oredered = torch.nn.functional.normalize(pred_identity_ordered, dim=-1)
        x = torch.mm(identity_vec_oredered, identity_vec_oredered.T)
        start = 0
        sim_loss = 0.
        dis_loss = 0.
        for end in cumsum_index:
            if start == end:
                continue
            sim_loss += torch.mean(1 - x[start:end, start:end])
            if end == x.shape[1]:
                pass
            else:
                dis_loss += torch.mean(torch.nn.ReLU()(x[start:end, end:]- 0.5)) 
            start = end
        return sim_loss+dis_loss


        
    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict[str, torch.Tensor]:
        loss_dict = super().get_loss_dict(outputs, batch, metrics_dict)
        self.epoch += 1
        if self.epoch > 20000 or self.epoch<5:
            loss = self.identity_loss_optimised(outputs["identity"], batch['image_idx'])
            loss_dict['identity_loss'] = loss
        return loss_dict


