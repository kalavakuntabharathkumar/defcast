import pandas as pd
import pytest
from pipeline.data_validation import validate_demand_frame, validate_image_manifest
from pipeline.make_demo_data import make_data
from pipeline.forecast import make_demo_demand, train_forecast
from pipeline.augmentation import validate_synthetic_images

def test_demo_manifest_validates(tmp_path):
    path=make_data(tmp_path,count=8); manifest=pd.read_csv(path)
    result=validate_image_manifest(manifest,tmp_path)
    assert result['rows']==8 and result['missing']==0

def test_duplicate_manifest_rejected(tmp_path):
    path=make_data(tmp_path,count=4); frame=pd.read_csv(path); frame.loc[1,'filename']=frame.loc[0,'filename']
    with pytest.raises(ValueError): validate_image_manifest(frame,tmp_path)

def test_bad_label_rejected(tmp_path):
    path=make_data(tmp_path,count=4); frame=pd.read_csv(path); frame.loc[0,'label']='unknown'
    with pytest.raises(ValueError): validate_image_manifest(frame,tmp_path)

def test_negative_demand_rejected():
    with pytest.raises(ValueError): validate_demand_frame(pd.DataFrame({'date':['2025-01-01'],'demand':[-1]}))

def test_forecast_metrics_are_finite():
    result=train_forecast(make_demo_demand(100))
    assert result['metrics']['test_rows']>0
    assert 0 <= result['metrics']['mape'] < 10

def test_augmentation_validates_image(tmp_path):
    manifest=make_data(tmp_path,count=2); frame=pd.read_csv(manifest)
    assert len(validate_synthetic_images([tmp_path/frame.iloc[0].filename]))==1
