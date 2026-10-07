import torch
import pytest
import models as m

@pytest.mark.parametrize('kernel,stride,pad',[(2,1,0),(3,2,1),(3,1,2)])
def test_im2col_fold_and_gradients_match_reference(kernel,stride,pad):
    torch.manual_seed(2);x=torch.randn(2,2,6,7,dtype=torch.float64,requires_grad=True)
    col=m.im2col(x,kernel,kernel,stride,pad)
    torch.testing.assert_close(col,torch.nn.functional.unfold(x,kernel,padding=pad,stride=stride))
    restored=m.col2im(col,x.shape,kernel,kernel,stride,pad)
    torch.testing.assert_close(restored,torch.nn.functional.fold(col,(6,7),kernel,padding=pad,stride=stride))
    assert torch.autograd.gradcheck(lambda a:m.im2col(a,kernel,kernel,stride,pad),
                                  (torch.randn(1,1,3,3,dtype=torch.float64,requires_grad=True),),fast_mode=True)

def test_convolution_forward_and_all_gradients_match_reference():
    torch.manual_seed(4);layer=m.Conv2d(2,3,3,stride=2,padding=1).double()
    x=torch.randn(2,2,7,6,dtype=torch.float64,requires_grad=True);z=x.detach().clone().requires_grad_()
    w=layer.weight.detach().clone().requires_grad_();b=layer.bias_param.detach().clone().requires_grad_()
    a=layer(x);ref=torch.nn.functional.conv2d(z,w,b,stride=2,padding=1)
    torch.testing.assert_close(a,ref)
    upstream=torch.randn_like(a);a.backward(upstream);ref.backward(upstream)
    for u,v in [(x.grad,z.grad),(layer.weight.grad,w.grad),(layer.bias_param.grad,b.grad)]:torch.testing.assert_close(u,v)

def test_pool_values_gradients_and_invalid_dimensions():
    torch.manual_seed(8);x=torch.randn(2,2,5,7,requires_grad=True);z=x.detach().clone().requires_grad_()
    a=m.MaxPool2d(2)(x);b=torch.nn.functional.max_pool2d(z,2)
    torch.testing.assert_close(a,b);a.sum().backward();b.sum().backward();torch.testing.assert_close(x.grad,z.grad)
    for args in [(0,3,3),(1,2,0),(1,2,3,0)]:
        with pytest.raises(ValueError):m.Conv2d(*args)
    with pytest.raises(ValueError):m.im2col(torch.ones(1,1,2,2),3,3)
