import torch
from torch.utils.mobile_optimizer import optimize_for_mobile
from models.pix2pix_model import Pix2PixModel
from options.test_options import TestOptions

opt = TestOptions().parse()
#opt.status = 'test'

device = torch.device('cpu')

model = Pix2PixModel(opt).netG
model.load_state_dict(torch.load("./checkpoints/CelebA-HQ_pretrained/latest_net_G.pth"))
model.eval()

#scripted_module = torch.jit.script(model)

rgb = torch.randn(1,3,256,256)
mask = torch.randn(1,19,256,256)
# obj_dic = {
    # '0': {'ACE': torch.randn(1,512)},
    # '1': {'ACE': torch.randn(1,512)},
    # '2': {'ACE': torch.randn(1,512)},
    # '3': {'ACE': torch.randn(1,512)},
    # '4': {'ACE': torch.randn(1,512)},
    # '5': {'ACE': torch.randn(1,512)},
    # '6': {'ACE': torch.randn(1,512)},
    # '7': {'ACE': torch.randn(1,512)},
    # '8': {'ACE': torch.randn(1,512)},
    # '9': {'ACE': torch.randn(1,512)},
    # '10': {'ACE': torch.randn(1,512)},
    # '11': {'ACE': torch.randn(1,512)},
    # '12': {'ACE': torch.randn(1,512)},
    # '13': {'ACE': torch.randn(1,512)},
    # '14': {'ACE': torch.randn(1,512)},
    # '15': {'ACE': torch.randn(1,512)},
    # '16': {'ACE': torch.randn(1,512)},
    # '17': {'ACE': torch.randn(1,512)},
    # '18': {'ACE': torch.randn(1,512)}
# }
#scripted_module = torch.jit.trace(model, (mask, rgb, obj_dic))
scripted_module = torch.jit.trace(model, (mask, rgb))


optimized_scripted_module = optimize_for_mobile(scripted_module)

# Export full jit version model (not compatible with lite interpreter)
scripted_module.save("generated/sean.pt")
# Export lite interpreter version model (compatible with lite interpreter)
scripted_module._save_for_lite_interpreter("generated/sean.ptl")

# using optimized lite interpreter model makes inference about 60% faster than the non-optimized lite interpreter model, which is about 6% faster than the non-optimized full jit model
optimized_scripted_module._save_for_lite_interpreter("app/src/main/assets/sean_scripted_optimized.ptl")
optimized_scripted_module.save("generated/sean_scripted_optimized2.pt")
