
import torch as t
from torch import nn
import transformers 
class BERT_Model(nn.Module):

    def __init__(
        self,x:t.Tensor, y:t.Tensor,num_attention_heads:int,
        hidden_dim:int, num_layers:int, last_activation:str,max_seq_length:int
    ) -> None:
        """
        Sets up BERT model based on huggingface implementation.
        Uses BertConfig to set up the BERT model and adds a classification head based on the class token.

        Args:
            x (torch.Tensor): Tensor of input data.
            y (torch.Tensor): Tensor of target data.
            num_attention_heads (int): Number of attention heads in the BERT model.
            hidden_dim (int): Hidden dimension for the BERT model.
            num_layers (int): Number of layers used of BERT model.
            last_activation (str): 'Softmax' or 'Sigmoid' activation for classification head.        
        """
        super().__init__()

        self.bert_config = transformers.BertConfig(
            vocab_size=x.shape[-1],
            hidden_size=hidden_dim,
            num_hidden_layers=num_layers,
            intermediate_size=hidden_dim,
            num_attention_heads =num_attention_heads,
            max_position_embeddings = max_seq_length # longest sequence possible
        )
        self.x_shape = x.shape
        self.out_shape = y.shape[-1] if y.ndim>1 else 1
        self.project = nn.Linear(x.shape[-1],hidden_dim,bias=False)#.to(x.device)        
        self.bert_model = transformers.BertModel(self.bert_config)#.to(x.device)
        self.linear = nn.Linear(hidden_dim,self.out_shape)#.to(x.device)
        
        self.final_activation =getattr(nn,last_activation)()

    def forward(self, x:t.Tensor):
        """Forward method for model.
        
        Args:
            x (torch.Tensor): Tensor of input data.

        Returns:
            Classification result (torch.Tensor)
        """
        position_ids = t.arange(0,x.shape[1],dtype=t.long,device=x.device)

        x_embedded = self.project(x)
        bert_out = self.bert_model(position_ids=position_ids ,inputs_embeds = x_embedded)
        #last_hidden_state = bert_out.last_hidden_state
        pooler_output = bert_out.pooler_output #uses special class token from last hidden state and feeds through dense and tanh
        #out = self.linear(last_hidden_state[:,-1,:])
        out = self.linear(pooler_output)
        return self.final_activation(out)

class LSTM_Model(nn.Module):

    def __init__(
        self,x:t.Tensor, y:t.Tensor, 
        hidden_dim:int, num_layers:int, 
        zero_initial_h_c:bool,last_activation:str
    ) -> None:
        
        """
        Sets up LSTM model based on PyTorch implementation.
        Adds a classification head based on last hidden state.

        Args:
            x (torch.Tensor): Tensor of input data.
            y (torch.Tensor): Tensor of target data.
            hidden_dim (int): Hidden dimension for the LSTM model.
            num_layers (int): Number of layers used of LSTM model.
            zero_initial_h_c (bool): True initialises h and c to zero, False to random.
            last_activation (str): 'Softmax' or 'Sigmoid' activation for classification head.        
        """

        super().__init__()
        self.x_shape =x.shape
        self.out_shape =  y.shape[-1] if y.ndim>1 else 1
        self.x_device = x.device
        self.hidden_dim =hidden_dim
        self.num_layers =num_layers
        self.zero_initial_h_c = zero_initial_h_c

        self.num_features = self.x_shape[-1]
        self.lstm = nn.LSTM(self.num_features,hidden_dim,num_layers,batch_first=True)
        self.linear = nn.Linear(hidden_dim,self.out_shape)
        self.final_activation =getattr(nn,last_activation)()

        self.h0_initial, self.c0_initial = self.make_initial_h0_c0(self.x_shape)

    def forward(self, x:t.Tensor):
        """Forward method for model.
        
        Args:
            x (torch.Tensor): Tensor of input data.

        Returns:
            Classification result (torch.Tensor)
        """
        if x.shape[0] != self.x_shape[0]:
            h0_initial, c0_initial = self.make_initial_h0_c0(x.shape)
        else:
            h0_initial, c0_initial = self.h0_initial, self.c0_initial

        out, (hn, cn) = self.lstm(x, (h0_initial, c0_initial))
        out = self.linear(out[:,-1,:])
        return self.final_activation(out)

    def make_initial_h0_c0(self, x_shape):
        if self.zero_initial_h_c:
            h0_initial = t.zeros(self.num_layers, x_shape[0], self.hidden_dim,device=self.x_device)
            c0_initial = t.zeros(self.num_layers, x_shape[0], self.hidden_dim,device=self.x_device)
        else:
            h0_initial = t.randn(self.num_layers, x_shape[0], self.hidden_dim,device=self.x_device)
            c0_initial = t.randn(self.num_layers, x_shape[0], self.hidden_dim,device=self.x_device)
        return h0_initial, c0_initial