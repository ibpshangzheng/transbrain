class Parameter_Class():

    def __init__(self,args) ->None:
        
        (self.train_species,self.train_model_type,self.train,self.device,self.trans_species,self.trans_model_type,self.transform_type,
         self.trans,self.gene_label,self.repeat_n,self.input_units,self.hidden_units1,self.hidden_units2,self.hidden_units5,self.output_units,self.weight_decay,
         self.max_epochs,self.learning_rate,self.total_steps,self.batch_size,self.loss,self.valid_ratio,self.modelname,self.use_bestrank,self.use_bestvalid,self.smoothfolder,
         self.individualtest,self.pretrain)=args

class Main_Class():

    def __init__(self,parameter_object) -> None:

        self.parameter_object=parameter_object

    def train(self):
            #遍历全部数据
            print('Total data num: ',len(mix_data_path_list))
            os.makedirs('./FinalModels_Cortex/'+parameter.modelname,exist_ok=True)
            shutil.move('./Succession_learning.log', './FinalModels_Cortex/'+parameter.modelname+'/Succession_learning.log')

            for data_id in range(len(mix_data_path_list)):
                mix_data_path = mix_data_path_list[data_id]
                print(mix_data_path)
                print('Loading data ...............................................')
                X,y,dfScLabels,genename=generate_human_matrix(mix_data_path,self.parameter_object)
                X = np.concatenate((X,y),axis=1)
                print('generate matrix done')
                print(np.unique(dfScLabels,return_index=True))

                data_label = mix_data_path.split('_')[-1].replace('.h5ad','')
                modelroot = './FinalModels_Cortex/'+parameter.modelname+'/Data_'+data_label
                os.makedirs(modelroot,exist_ok=True)
                
                for i in tqdm(range(self.parameter_object.repeat_n)):
                    print('Now repeat: ',i)
                    print('-------------------------------------------------------')

                    train_data, valid_data = train_test_split(X, test_size=parameter.valid_ratio)
                    print(type(train_data))
                    # train_DNN(self.parameter_object,train_data,valid_data,i,mix_data_path) 


    def trans(self):
        GeneWeights(self.parameter_object)
        Transform_Human_Mouse(self.parameter_object)
        Multiclass_TransformtoMouse(self.parameter_object)
    

    def run(self):

        if self.parameter_object.train:
            self.train()
        if self.parameter_object.trans:
            self.trans()
