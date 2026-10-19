import matplotlib.pyplot as plt
import pandas as pd


softmax=pd.read_csv("softmax.csv",index_col=0)

fig,axes = plt.subplots(nrows=1,ncols=2,figsize = (14,6))
type=['softmax_bndo','softmax_bn','softmax_do02','softmax_do05','softmax_do08','softmax_noBD']
fig.suptitle("softmax with/witohut DropOut/BatchNormalization",size = 'x-large')

for i in range(len(type)):
    data=softmax[(softmax['type']==type[i])]
    axes[0].plot(data['step'],data['train_loss'],label = "{}".format(type[i]))
    axes[0].set_xlabel('step',size = 'x-large')
    axes[0].set_ylabel("train_loss",size = 'x-large')
    #axes[0].set_ylim( (softmax['train_loss']).min() ,  (softmax['train_loss']).max() )
    axes[0].legend()

    axes[1].plot(data['step'],data['train_acc'],label = "{}".format(type[i]))
    axes[1].set_xlabel('step',size = 'x-large')
    axes[1].set_ylabel("train_accuracy",size = 'x-large')
    #axes[1].set_ylim( (softmax['train_acc']).min(),  (softmax['train_acc']).max())
    axes[1].legend()
    
fig.tight_layout()



mlp=pd.read_csv("mlp.csv",index_col=0)

fig,axes = plt.subplots(nrows=1,ncols=2,figsize = (14,6))
type=['mlp_bndo','mlp_bn','mlp_do02','mlp_do05','mlp_do08','mlp_noBD']
fig.suptitle("mlp with/witohut DropOut/BatchNormalization",size = 'x-large')

for i in range(len(type)):
    data=mlp[(mlp['type']==type[i])]
    axes[0].plot(data['step'],data['train_loss'],label = "{}".format(type[i]))
    axes[0].set_xlabel('step',size = 'x-large')
    axes[0].set_ylabel("train_loss",size = 'x-large')
    #axes[0].set_ylim( (mlp['train_loss']).min() - .001 ,  (mlp['train_loss']).max() + .001 )
    axes[0].legend()

                     
    axes[1].plot(data['step'],data['train_acc'],label = "{}".format(type[i]))
    axes[1].set_xlabel('step',size = 'x-large')
    axes[1].set_ylabel("train_accuracy",size = 'x-large')
    #axes[1].set_ylim( (mlp['train_accuracy']).min(),  (mlp['train_acc']).max())
    axes[1].legend()
    
fig.tight_layout()



cnn = pd.read_csv("CNN.csv",index_col=0)

fig,axes = plt.subplots(nrows=1,ncols=2,figsize = (14,6))
type=['cnn_bndo','cnn_bn','cnn_do02','cnn_do05','cnn_do08','cnn_noBD']
fig.suptitle("CNN with/witohut DropOut/BatchNormalization",size = 'x-large')

for i in range(len(type)):
    data=cnn[(cnn['type']==type[i])]
    axes[0].plot(data['step'],data['train_loss'],label = "{}".format(type[i]))
    axes[0].set_xlabel("step",size = 'x-large')
    axes[0].set_ylabel("train_loss",size = 'x-large')
    axes[0].legend()
    
    axes[1].plot(data['step'],data['train_acc'],label = "{}".format(type[i]))
    axes[1].set_xlabel("step",size = 'x-large')
    axes[1].set_ylabel("train_accuracy",size = 'x-large')
    axes[1].legend()
    
fig.tight_layout()
plt.show()    


# compare three classifier with BN

fig,axes = plt.subplots(nrows=1,ncols=2,figsize = (14,6))
compare_type=['softmax_bn','mlp_bn','cnn_bn']
fig.suptitle("softmax MLP CNN with BN ",size = 'x-large')

data_cnn=cnn[(cnn['type']=='cnn_bn')]
data_sfm=softmax[(softmax['type']=='softmax_bn')]
data_mlp=mlp[(mlp['type']=='mlp_bn')]
compare_data=[data_cnn,data_sfm,data_mlp]

axes[0].plot(data_cnn['step'],data_cnn['train_loss'],label = 'cnn_bn')
axes[0].plot(data_sfm['step'],data_sfm['train_loss'],label = 'softmax_bn')
axes[0].plot(data_mlp['step'],data_mlp['train_loss'],label = 'mlp_bn')
axes[0].set_xlabel("step",size = 'x-large')
axes[0].set_ylabel("train_loss",size = 'x-large')
axes[0].legend()
    
axes[1].plot(data_cnn['step'],data_cnn['train_acc'],label = 'cnn_bn')
axes[1].plot(data_sfm['step'],data_sfm['train_acc'],label = 'softmax_bn')
axes[1].plot(data_mlp['step'],data_mlp['train_acc'],label = 'mlp_bn')
axes[1].set_xlabel("step",size = 'x-large')
axes[1].set_ylabel("train_accuracy",size = 'x-large')
axes[1].legend()

fig.tight_layout()
plt.show()    
