import torch
from torch.utils.mobile_optimizer import optimize_for_mobile
from models.pix2pix_model import Pix2PixModel
from options.test_options import TestOptions
from models.networks.architecture import Zencoder

 
opt = TestOptions().parse()

model = Pix2PixModel(opt).netG
model.load_state_dict(torch.load("./checkpoints/CelebA-HQ_pretrained/latest_net_G.pth"), strict=False)
model.eval()

mask = torch.randn(1,19,256,256)
rgb = torch.randn(1,3,256,256)
rgb_m = torch.randn(1,19,256,256)
#style_codes = torch.randn(1,19,512)

scripted_module = torch.jit.trace(model, (mask, rgb, rgb_m))
#scripted_module = torch.jit.script(model)

optimized_scripted_module = optimize_for_mobile(scripted_module)


# Export full jit version model (not compatible with lite interpreter)
#scripted_module.save("generated/sean.pt")
# Export lite interpreter version model (compatible with lite interpreter)
scripted_module._save_for_lite_interpreter("app/src/main/assets/sean.ptl")
# using optimized lite interpreter model makes inference about 60% faster than the non-optimized lite interpreter model, which is about 6% faster than the non-optimized full jit model
#optimized_scripted_module._save_for_lite_interpreter("generated/sean_scripted_optimized.ptl")


# Creating model for Zencoder
#scripted_module_zenc = torch.jit.trace(model, (mask, rgb, style_codes))
#optimized_scripted_module_zenc = optimize_for_mobile(scripted_module_zenc)
#scripted_module_zenc._save_for_lite_interpreter("generated/sean_zenc.ptl")