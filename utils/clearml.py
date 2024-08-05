from clearml import Task, OutputModel, StorageManager
from utils.singleton import Singleton

import os
class ClearMLManager(metaclass=Singleton):

    def __init__(self, projectName, taskName):
        
        self.projectName = projectName
        self.taskName = taskName
        self.apiHost = os.getenv('CLEARML_API_HOST')
        self.webHost = os.getenv('CLEARML_WEB_HOST')
        self.filesHost = os.getenv('CLEARML_FILES_HOST')
        Task.set_credentials(
            api_host= self.apiHost,
            web_host= self.webHost,
            files_host= self.filesHost,
            key=os.getenv('CLEARML_KEY'),
            secret=os.getenv('CLEARML_SECRET'),
            store_conf_file=True
            
        )
        self._task = Task.init(project_name=self.projectName, task_name=self.taskName, output_uri = self.filesHost)
        # self._task.register_abort_callback(shutDownServer(), callback_execution_timeout=30.0)


        self._logger = self._task.get_logger()
        self._logger.set_default_upload_destination(uri=self.filesHost)
        self._logger.tensorboard_auto_group_scalars(group_scalars=False)
        self._outputModel = OutputModel(task=self.getTask(), tags = ['production']) #may i need to use VE for tags

        # self._task.update_output_model(model_path = f'{self.filesHost}/{self.projectName}/{self.taskName}.{self._task.id}/artifacts/weights/model.onnx')
        self.modelPath = ''

    def setMetadata(self , key,value):
        self._outputModel.set_metadata(key, value, v_type=None)
    
    # Task
    def uploadArtifacts(self, name, object, metadata = None):
        self._task.upload_artifact(name = name,artifact_object = object, metadata = metadata)


    def getTask(self):
        return self._task.current_task()

    def getTasks(self):
        return self._task.get_tasks(project_name = self.projectName, task_name = self.taskName,task_filter={'status': ['completed']})
    
    def getTaskById(self, id):
        return self._task.get_task(task_id = id)
    
    def connectConfig(self, config, name):
        return self._task.connect_configuration(config, name=name)
    
    def close(self, status:str = 'Ok', action = "completed"):
        if action not in ["completed", "failed"]:
            raise ValueError("Invalid action. Action must be 'completed' or 'failed'.")
        if action == "completed":
            self._task.close()
        elif action == "failed":
            self._task.close()
            self._task.mark_failed(status_message = status, force=True )
        # self._task.close

    def getLabelStats(self):
        return self._task.labels_stats()
    
    def getNumberOfClasses(self,task = None):
        if task is None:
            return self._task.get_num_of_classes()
        else:
            return task.get_num_of_classes()
    
    def getModelLabels(self):
        return self._task.get_labels_enumeration()
    
    def updateModelLabels(self, y, yMapped):
        labels = {f'"{key}"': int(value) for key, value in  zip(y, yMapped)}
        self._task.connect_label_enumeration(labels)
    
    def getArtifacts(self, task):
        return task.artifacts
    def getBestModel(self, currentNumClasses):
        tasks = self.getTasks()
        bestAccuracy = -1
        bestTask = None
        
        for task in tasks:
            numClasses = self.getNumberOfClasses(task)
            metrics = task.get_last_scalar_metrics()
            
            # Si la clé 'Summary' existe dans les métriques et contient 'test_accuracy'
            if 'Summary' in metrics and 'test_accuracy' in metrics['Summary']:
                
                accuracy = metrics['Summary']['test_accuracy']['last']
            else:
                accuracy = 0.0
            
            if numClasses == currentNumClasses:
                if accuracy > bestAccuracy:
                    bestAccuracy = accuracy
                    bestTask = task.id
        
        if bestTask is not None:
            return {'task_id': bestTask, 'accuracy': bestAccuracy}
        else:
            return {'task_id': None, 'accuracy': None}


    def getConfig(self, name,task = None):
        if task is None:
            return self._task.get_configuration_object_as_dict(name=name)
        else : 
            return task.get_configuration_object_as_dict(name=name)



    def updateModelLabels(self, y, yMapped):
        labels = {f'"{key}"': int(value) for key, value in  zip(y, yMapped)}
        self._task.connect_label_enumeration(labels)

    
    def getModelPath(self):
        return self.modelPath
    
    

    # Storage

    def getCopy(self, url):
        return StorageManager.get_local_copy(remote_url = url)
    
    # Logs
    def getLogger(self):
        return self._logger
    
    def reportScalar(self, title, series, value, iteration):
        self._logger.report_scalar(title, series, value, iteration)

    def reportValue(self, name, value):
        self._logger.report_single_value(name,value)
        

    # Dataset
    
    def addFileToDataset(self, path:str):
        return self._dataset.add_files(path)
    
    def setDatasetMetadatas(self, df):
        return self._dataset.set_metadata(df)
    
