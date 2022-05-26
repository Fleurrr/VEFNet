#336
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.models.vgg import model_urls
from torchvision import models
import numpy as np
import time
import sys
from collections import OrderedDict
import scipy.io
import os
import model.netvlad as netvlad

class VFENet(nn.Module):
     def __init__(self, model_path=None, pretrained=True, num_class = 100, freeze = True, pooling = 'wpca', mode = 'train'):
         super(VFENet, self).__init__()
         self.num_class = num_class
         self.freeze = freeze
         self.pooling = pooling
         model_urls['vgg16_bn'] = model_urls['vgg16_bn'].replace('https://', 'http://')
         vgg_pretrained_features = models.vgg16_bn(pretrained=pretrained).features
         self.layers = nn.Sequential(OrderedDict([  
                    ('conv1', nn.Sequential(vgg_pretrained_features[0], vgg_pretrained_features[1], vgg_pretrained_features[2],)),
                    ('conv2', nn.Sequential(vgg_pretrained_features[3], vgg_pretrained_features[4], vgg_pretrained_features[5], vgg_pretrained_features[6],)),
                    ('conv3', nn.Sequential(vgg_pretrained_features[7], vgg_pretrained_features[8], vgg_pretrained_features[9],)),
                    ('conv4', nn.Sequential(vgg_pretrained_features[10], vgg_pretrained_features[11], vgg_pretrained_features[12], vgg_pretrained_features[13],)),
                    ('conv5', nn.Sequential(vgg_pretrained_features[14], vgg_pretrained_features[15], vgg_pretrained_features[16], )),
                    ('conv6', nn.Sequential(vgg_pretrained_features[17], vgg_pretrained_features[18], vgg_pretrained_features[19], )),
                    ('conva', nn.Sequential(vgg_pretrained_features[20], vgg_pretrained_features[21], vgg_pretrained_features[22], vgg_pretrained_features[23],)),
                    ('convb', nn.Sequential(vgg_pretrained_features[24], vgg_pretrained_features[25], vgg_pretrained_features[26],)),
                    ('conv8', nn.Sequential(vgg_pretrained_features[27], vgg_pretrained_features[28], vgg_pretrained_features[29],)),
                    ('conv9', nn.Sequential(vgg_pretrained_features[30], vgg_pretrained_features[31], vgg_pretrained_features[32], vgg_pretrained_features[33],)),
                    ('conv10', nn.Sequential(vgg_pretrained_features[34], vgg_pretrained_features[35], vgg_pretrained_features[36],)),
                    ('fc3',  nn.Sequential(nn.Linear(336, 256), nn.ReLU())),
                    ('fc5',   nn.Sequential(nn.Dropout(0.5), nn.Linear(256, 256), nn.ReLU())),
                    ]))
         self.global_pool = nn.AdaptiveAvgPool2d((1,1))
         if self.freeze:       
             for param in self.layers.conv1.parameters():
                 param.requires_grad= False
             for param in self.layers.conv2.parameters():
                 param.requires_grad= False
             for param in self.layers.conv3.parameters():
                 param.requires_grad= False
             for param in self.layers.conv4.parameters():
                 param.requires_grad= False
                                                  
         if self.pooling == 'cls':           
             self.out_pred = nn.Sequential(nn.Linear(262144, self.num_class))
         elif self.pooling == 'avg':
             self.avg_pool = nn.AdaptiveAvgPool2d((32, 32))
         elif self.pooling == 'branch':
             self.branch = nn.Sequential(nn.Dropout(0.5), nn.Linear(256 * 2, 2))
         elif self.pooling == 'wpca':
             self.wpca1d = nn.Sequential(nn.Conv1d(512, 32, kernel_size = 8))
         elif self.pooling == 'netvlad':
             self.net_vlad = netvlad.NetVLAD(num_clusters = 32, dim = 256, vladv2=False)
             print('Init Done!')
         ##########################################################################
         ####     layers for Cross-Modality Transformer module (CrossAtt)      ####
         ##########################################################################
         self.fc_vis     = nn.Sequential(nn.Linear(256, 256), nn.ReLU())
         self.fc_baseVec = nn.Sequential(nn.Linear(256, 256), nn.ReLU())
         self.fc_event   = nn.Sequential(nn.Linear(256, 256), nn.ReLU())
         self.fc_hiddenState_vis   = nn.Sequential(nn.Linear(256, 1), nn.ReLU())
         self.fc_hiddenState_event = nn.Sequential(nn.Linear(256, 1), nn.ReLU())
        
         ##########################################################################
         ####            layers for Transformer module (SelfAtt)               ####
         ##########################################################################        
         self.transformer_ff_vis   = nn.Sequential(nn.Linear(256, 256), nn.ReLU()) 
         self.transformer_ff_event = nn.Sequential(nn.Linear(256, 256), nn.ReLU()) 

         self.selfAtt_key_vis   = nn.Sequential(nn.Linear(256, 256), nn.ReLU()) 
         self.selfAtt_query_vis = nn.Sequential(nn.Linear(256, 256), nn.ReLU()) 
         self.selfAtt_value_vis = nn.Sequential(nn.Linear(256, 256), nn.ReLU())  

         self.selfAtt_key_event   = nn.Sequential(nn.Linear(256, 256), nn.ReLU()) 
         self.selfAtt_query_event = nn.Sequential(nn.Linear(256, 256), nn.ReLU()) 
         self.selfAtt_value_event = nn.Sequential(nn.Linear(256, 256), nn.ReLU())  

         self.selfAtt_fuse_vis   = nn.Sequential(nn.Linear(256, 256), nn.ReLU()) 
         self.selfAtt_fuse_event = nn.Sequential(nn.Linear(256, 256), nn.ReLU())
          
     
     def encode_layer(self, x ,in_layer='conv1', out_layer='conv10'):
         run = False
         for name, module in self.layers.named_children():
             if name == in_layer:
                 run = True
             if run:
                 x = module(x)
                 if name == out_layer:
                     return x
                     
     def samp_layer(self, x ,in_layer='fc3', out_layer='fc5'):
         run = False
         for name, module in self.layers.named_children():
             if name == in_layer:
                 run = True
             if run:
                 x = module(x)
                 if name == out_layer:
                     return x
                     
     def samp_layer_s(self, x_vis, x_event):
         return self.global_pool(x_vis), self.global_pool(x_event)
     
     def fusion_layer(self, x_vis, x_event, useCrossAtt=False, useSelfAtt=False):
         if useCrossAtt: 
             #### Cross-Attention Layers 
             base_vector_m0 = torch.mul(x_vis, x_event)  ## base_vector_m0: torch.Size([1568, 256]) 

             hiddenState_v = torch.mul(torch.tanh(self.fc_vis(x_vis)), torch.tanh(self.fc_baseVec(base_vector_m0)))
             atten_weight_v = torch.transpose(F.softmax(torch.transpose(self.fc_hiddenState_vis(hiddenState_v), 1, 2), dim=1), 1, 2)
             
             # atten_weight_v.shape: torch.Size([1, 256])
             atten_weight_v_matrix = atten_weight_v.expand(x_vis.size(0), x_vis.size(1), base_vector_m0.size(2))
                          
                          
             # print("==>> atten_weight_v_matrix.shape: ", atten_weight_v_matrix.shape) 
             x_vis_atteded = torch.tanh(torch.mul(atten_weight_v_matrix, x_vis))
             x_vis = x_vis + x_vis_atteded
             
             hiddenState_e = torch.mul(torch.tanh(self.fc_event(x_event)), torch.tanh(self.fc_baseVec(base_vector_m0))) 
             atten_weight_e = torch.transpose(F.softmax(torch.transpose(self.fc_hiddenState_event(hiddenState_e), 1, 2), dim=1), 1, 2)

             atten_weight_e_matrix = atten_weight_e.expand(x_vis.size(0), x_vis.size(1), base_vector_m0.size(2))
             x_event_atteded = torch.tanh(torch.mul(atten_weight_e_matrix, x_event)) 
             x_event = x_event + x_event_atteded
             
            
         if useSelfAtt: 
             #### Self-Attention Layers 
             x_vis_temp = torch.mul(self.selfAtt_key_vis(x_vis), self.selfAtt_query_vis(x_vis))
             x_vis_temp = torch.mul(F.softmax(x_vis_temp, dim=1), self.selfAtt_value_vis(x_vis))
             x_vis = x_vis + self.selfAtt_fuse_vis(x_vis_temp)


             x_event_temp = torch.mul(self.selfAtt_key_event(x_event), self.selfAtt_query_event(x_event)) 
             x_event_temp = torch.mul(F.softmax(x_event_temp, dim=1), self.selfAtt_value_event(x_event))
                            
             x_event = x_event + self.selfAtt_fuse_event(x_event_temp)
         return x_vis, x_event
    
     def ouput_layer(self, x_vis, x_event, out_layer = 'fc6'):
         if out_layer == 'fc6':
             #x = torch.cat((x_vis.unsqueeze(2), x_event.unsqueeze(2)), axis=2) #torch.Size([24, 256, 2, 256])
             x = torch.mul(x_vis, x_event)
             x = F.normalize(x, p=2, dim=1)
             return x 
         elif out_layer == 'fc6_softmax':
             x = x_vis + x_event 
             return F.softmax(x)
         elif out_layer == 'fc6_pred':
             x = torch.cat([x_vis, x_event], axis = 2)
             return self.out_pred(x.reshape(x.shape[0], x.shape[1]*x.shape[2]))
         elif out_layer == 'fc6_pool':
             x = torch.mul(x_vis, x_event)
             x = self.avg_pool(x)
             return x.reshape(x.shape[0], x.shape[1]*x.shape[2])
         elif out_layer == 'fc6_fc':
             x_vis = x_vis.transpose(2,1)
             x_event = x_event.transpose(2,1)
             x = self.branch(torch.cat((x_vis, x_event), 2)) 
             return x.reshape(x.shape[0], x.shape[1]*x.shape[2])
         elif out_layer == 'fc6_wpca':
             x = torch.cat([x_vis, x_event], axis = 2)
             x = torch.transpose(x, 1, 2)
             x = self.wpca1d(x)
             x = x.reshape(x.shape[0], x.shape[1]* x.shape[2])
             return x
         elif out_layer == 'fc6_netvlad':
             #x = torch.cat((x_vis.unsqueeze(2), x_event.unsqueeze(2)), axis=2)
             x = torch.mul(x_vis, x_event)
             x = x.reshape(x.shape[0], x.shape[1], 16, 16)
             return self.net_vlad(x)
                           
if __name__ == "__main__": 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VFENet().to(device)
    
    input_vis = torch.cuda.FloatTensor(12, 3, 320, 160)
    input_event = torch.cuda.FloatTensor(12, 3, 320, 160)
    
    vis_ec = (model.encode_layer(input_vis))
    event_ec = (model.encode_layer(input_event))
    
    B, C, H, W = vis_ec.shape[0], vis_ec.shape[1], vis_ec.shape[2], vis_ec.shape[3]
    vis_ec = vis_ec.reshape(B, C, H*W)
    event_ec = event_ec.reshape(B, C, H*W)
    
    vis_ec, event_ec = model.samp_layer(vis_ec), model.samp_layer(event_ec)
    fus_vec = model.fusion_layer(vis_ec, event_ec, useCrossAtt=True, useSelfAtt=True)
    output = model.ouput_layer(fus_vec, out_layer = 'fc6_pred')
    