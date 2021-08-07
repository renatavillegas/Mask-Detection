from imageai.Detection.Custom import DetectionModelTrainer

trainer = DetectionModelTrainer()
trainer.setModelTypeAsYOLOv3()
trainer.setDataDirectory(data_directory="Dataset")

trainer.setTrainConfig(object_names_array=["with_mask without_mask"], batch_size=4, num_experiments=20, 
                       )

trainer.trainModel()