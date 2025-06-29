from config_reader import ConfigurationReader

conf = ConfigurationReader("/le_thanh_van_118/workspace/hiep_workspace/model_params.json")

print(conf)
print(conf.prediction)
print(conf.reduction)
