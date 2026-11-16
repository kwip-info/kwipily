"""Editable encoder/decoder composition agrees with the legacy module after
explicit weight transfer. Fresh independent initialization is NOT identical.
"""
import pytest
import torch
from runner.attention_execution import create_attention_module, execute_attention
from runner.attention_configuration import normalize_attention, configured_input_ports

@pytest.mark.parametrize('norm_first,bias', [(False,True),(True,True),(True,False)])
def test_encoder_decoder_container_preserves_values_gradients_and_mask_roles(norm_first,bias):
    full_config=normalize_attention('transformer',dict(d_model=4,nhead=2,dim_feedforward=7,num_encoder_layers=2,num_decoder_layers=2,dropout=0.,norm_first=norm_first,bias=bias,source_mask_mode='supplied',target_mask_mode='causal',memory_mask_mode='supplied',source_padding_mode='supplied',target_padding_mode='supplied',memory_padding_mode='supplied'))
    shared={key:full_config[key] for key in ('d_model','nhead','dim_feedforward','dropout','activation','layer_norm_eps','norm_first','bias')}
    enc_config=normalize_attention('transformer_encoder',dict(shared,num_layers=2,final_norm=True,mask_mode='supplied',padding_mask_mode='supplied'))
    dec_config=normalize_attention('transformer_decoder',dict(shared,num_layers=2,final_norm=True,target_mask_mode='causal',memory_mask_mode='supplied',target_padding_mode='supplied',memory_padding_mode='supplied'))
    torch.manual_seed(7)
    full=create_attention_module('transformer',full_config)
    enc=create_attention_module('transformer_encoder',enc_config)
    dec=create_attention_module('transformer_decoder',dec_config)
    enc.load_state_dict(full.encoder.state_dict(),strict=True)
    dec.load_state_dict(full.decoder.state_dict(),strict=True)
    full.train();enc.train();dec.train()
    source=torch.randn(2,3,4,requires_grad=True);target=torch.randn(2,2,4,requires_grad=True)
    copied_source=source.detach().clone().requires_grad_();copied_target=target.detach().clone().requires_grad_()
    src_mask=torch.zeros(2,3,3,dtype=torch.int64);src_mask[:,0,1]=1
    memory_mask=torch.zeros(2,2,3,dtype=torch.int64);memory_mask[:,1,0]=1
    src_padding=torch.zeros(2,3,dtype=torch.int64)
    target_padding=torch.zeros(2,2,dtype=torch.int64)
    memory_padding=torch.zeros(2,3,dtype=torch.int64);memory_padding[:,2]=1
    sources=dict(source=source,target=target,src_mask=src_mask,src_key_padding_mask=src_padding,memory_mask=memory_mask,tgt_key_padding_mask=target_padding,memory_key_padding_mask=memory_padding)
    expected=execute_attention('transformer',full_config,[sources[p] for p in configured_input_ports('transformer',full_config)],full)['out']
    memory=execute_attention('transformer_encoder',enc_config,[copied_source,src_mask,src_padding],enc)['out']
    actual=execute_attention('transformer_decoder',dec_config,[copied_target,memory,memory_mask,target_padding,memory_padding],dec)['out']
    torch.testing.assert_close(actual,expected,atol=2e-6,rtol=2e-5)
    probe=torch.arange(actual.numel(),dtype=torch.float32).reshape_as(actual)/actual.numel()
    (actual*probe).sum().backward();(expected*probe).sum().backward()
    torch.testing.assert_close(copied_source.grad,source.grad,atol=2e-6,rtol=2e-5)
    torch.testing.assert_close(copied_target.grad,target.grad,atol=2e-6,rtol=2e-5)
    for prefix,member in [('encoder',enc),('decoder',dec)]:
        original=dict(full.named_parameters())
        for name,parameter in member.named_parameters():
            torch.testing.assert_close(parameter.grad,original[prefix+'.'+name].grad,atol=2e-6,rtol=2e-5)
