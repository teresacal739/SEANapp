import torch
from torch.utils.mobile_optimizer import optimize_for_mobile
from models.pix2pix_model import Pix2PixModel
from options.test_options import TestOptions
from models.networks.architecture import Zencoder


# class ModelSEANmobile(torch.nn.Module):
    # def __init__(self, opt):
        # super().__init__()
        # self.opt = opt
        # self.model = Pix2PixModel(opt).netG
        # self.model.load_state_dict(torch.load("./checkpoints/CelebA-HQ_pretrained/latest_net_G.pth"))
        # self.style_encoder = Zencoder(3,512)    
    # def forward(self, mask, rgb, rgb_mask):
        # x = self.style_encoder(input=rgb, segmap=rgb_mask)
        # res = self.model(mask, rgb, obj_dic=x)
        # return res
    
opt = TestOptions().parse()

model = Pix2PixModel(opt).netG
model.load_state_dict(torch.load("./checkpoints/CelebA-HQ_pretrained/latest_net_G.pth"), strict=False)
model.eval()

#model2 = Pix2PixModel(opt).netG.Zencoder
#model2.load_state_dict(torch.load("./checkpoints/CelebA-HQ_pretrained/latest_net_G.pth"), strict=False)
#model2.eval()
#style_enc = Zencoder(3,512)
#scripted_module = torch.jit.script(model)

#model = ModelSEANmobile(opt)
#model.eval()


rgb = torch.randn(1,3,256,256)
mask = torch.randn(1,19,256,256)
style_code = torch.randn(1,19,512)
scripted_module = torch.jit.trace(model, (mask, rgb, style_code))

#scripted_module2 = torch.jit.trace(model2, (rgb, mask_rgb))

#scripted_module = torch.jit.script(model)


optimized_scripted_module = optimize_for_mobile(scripted_module)


# Export full jit version model (not compatible with lite interpreter)
#scripted_module.save("generated/sean.pt")
# Export lite interpreter version model (compatible with lite interpreter)
#scripted_module._save_for_lite_interpreter("app/src/main/assets/sean.ptl")

scripted_module._save_for_lite_interpreter("app/src/main/assets/sean_enc.ptl")

#torch._save_for_lite_interpreter({'model': scripted_module, 'model2': scripted_module2}, "app/src/main/assets/sean_enc.ptl")

# using optimized lite interpreter model makes inference about 60% faster than the non-optimized lite interpreter model, which is about 6% faster than the non-optimized full jit model
#optimized_scripted_module._save_for_lite_interpreter("generated/sean_scripted_optimized.ptl")
#optimized_scripted_module.save("generated/sean_scripted_optimized2.pt")
