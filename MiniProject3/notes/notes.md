### model_new_no_norm 
vae = VAE(latent_dim=16, hidden_dim=256, x_dim=3072).to(device)
VAE(
  (encoder): Encoder(
    (fc_1): Linear(in_features=3072, out_features=2048, bias=True)
    (bn1): BatchNorm1d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (fc_2): Linear(in_features=2048, out_features=1024, bias=True)
    (bn2): BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (fc_3): Linear(in_features=1024, out_features=256, bias=True)
    (bn3): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (LeakyReLU): LeakyReLU(negative_slope=0.2)
    (dropout): Dropout(p=0.1, inplace=False)
    (fc_mean): Linear(in_features=256, out_features=16, bias=True)
    (fc_var): Linear(in_features=256, out_features=16, bias=True)
  )
  (decoder): Decoder(
    (fc_1): Linear(in_features=16, out_features=256, bias=True)
    (bn1): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (fc_2): Linear(in_features=256, out_features=1024, bias=True)
    (bn2): BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (fc_3): Linear(in_features=1024, out_features=2048, bias=True)
    (bn3): BatchNorm1d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (fc_5): Linear(in_features=2048, out_features=3072, bias=True)
    (LeakyReLU): LeakyReLU(negative_slope=0.2)
  )
)
fretchet_distance = 14.548020485690671

loss_train =  13840.035757211539,

learning curves:
![alt text](image.png)
![alt text](image-1.png)

### simpler_net_no_norm

fretchet_distance  = 11.389373996045563
loss_train = 12930.3
VAE(
  (encoder): Encoder(
    (fc_1): Linear(in_features=3072, out_features=1024, bias=True)
    (bn1): BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (fc_2): Linear(in_features=1024, out_features=256, bias=True)
    (bn2): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (LeakyReLU): LeakyReLU(negative_slope=0.2)
    (fc_mean): Linear(in_features=256, out_features=16, bias=True)
    (fc_var): Linear(in_features=256, out_features=16, bias=True)
  )
  (decoder): Decoder(
    (fc_1): Linear(in_features=16, out_features=256, bias=True)
    (bn1): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (fc_2): Linear(in_features=256, out_features=1024, bias=True)
    (bn2): BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (fc_5): Linear(in_features=1024, out_features=3072, bias=True)
    (LeakyReLU): LeakyReLU(negative_slope=0.2)
  )
)

loss_train = 12759.47805417239
loss_val = 13187.054415343238
fretchet_distance = 5.402452784140678
![alt text](image-4.png)

![alt text](image-5.png)
![alt text](image-3.png)


### simpler_net_no_batch_norm_no_norm

VAE(
  (encoder): Encoder(
    (fc_1): Linear(in_features=3072, out_features=1024, bias=True)
    (fc_2): Linear(in_features=1024, out_features=256, bias=True)
    (LeakyReLU): LeakyReLU(negative_slope=0.2)
    (fc_mean): Linear(in_features=256, out_features=16, bias=True)
    (fc_var): Linear(in_features=256, out_features=16, bias=True)
  )
  (decoder): Decoder(
    (fc_1): Linear(in_features=16, out_features=256, bias=True)
    (fc_2): Linear(in_features=256, out_features=1024, bias=True)
    (fc_5): Linear(in_features=1024, out_features=3072, bias=True)
    (LeakyReLU): LeakyReLU(negative_slope=0.2)
  )
)

train = 12665.220306061126,
loss val = 12992.720318903688
fretchet_distance = 5.127452403743504
![alt text](image-6.png)
![alt text](image-7.png)
![alt text](image-8.png)