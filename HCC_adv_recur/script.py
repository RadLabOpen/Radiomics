def n4_bias_correction(input_image, iterations=25, fitting_level=5):
    try:
        mask_image = sitk.OtsuThreshold(input_image, 0, 1, 200)
        input_image = sitk.Cast(input_image, sitk.sitkFloat32)
        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        corrector.SetMaximumNumberOfIterations([iterations])
        corrector.SetNumberOfControlPoints([fitting_level] * input_image.GetDimension())
        corrector.SetConvergenceThreshold(0.001)
        output_image = corrector.Execute(input_image, mask_image)
        return output_image
    except Exception as e:
        print(f"Bias correction 실패: {e}")
        return None

def process_file(nrrd_file_path, output_file_path, fitting_level):
    try:
        nrrd_image = sitk.ReadImage(nrrd_file_path)
        corrected_image = n4_bias_correction(nrrd_image, iterations=25, fitting_level=fitting_level)
        if corrected_image is not None:
            sitk.WriteImage(corrected_image, output_file_path)
            return print(output_file_path)
        else:
            print(f"{output_file_path} 처리 중 오류 발생: Bias correction 실패")
            return None
    except Exception as e:
        print(f"{output_file_path} 처리 중 오류 발생: {e}")
        return None



def Run_n4_correction(input_folder, output_folder, fitting_level):
    filelist = glob.glob(input_folder+r'\1_*')+glob.glob(input_folder+r'\2_*')+glob.glob(input_folder+r'\3_*')+glob.glob(input_folder+r'\4_*')+glob.glob(input_folder+r'\5_*')+glob.glob(input_folder+r'\6_*')+glob.glob(input_folder+r'\7_*')
    for input_path in filelist:
        output_path= os.path.join(output_folder, os.path.basename(input_path))
        process_file(input_path,output_path, fitting_level)
        

input_folder = r'E:\2411radiomics\Raw'
output_folder = r'E:\2411radiomics\N4corrected'
fitting_level = 5

Run_n4_correction(input_folder, output_folder, fitting_level)

def pyradiomics_feature_extractor(args):
    image_path, mask_path, save_path = args
    # load the nifti files and mask files
    image = sitk.ReadImage(image_path)
    mask = sitk.ReadImage(mask_path)
    
    mask.SetDirection(image.GetDirection())
    mask.SetOrigin(image.GetOrigin())
    sitk.WriteImage(mask, mask_path, True)

    # Get the PyRadiomics logger (default log-level = INFO)
    logger = radiomics.logger
    logger.setLevel(logging.DEBUG)  # set level to DEBUG to include debug log messages in log file

    # Set up the handler to write out all log entries to a file
    handler = logging.FileHandler(filename='testLog.txt', mode='w')
    formatter = logging.Formatter("%(levelname)s:%(name)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # extrator setting
    settings = {}
    settings['binCount'] = 16
    settings['resampledPixelSpacing'] = [1,1,1]  
    #settings['interpolator'] = sitk.sitkBSpline
    settings['normalize'] = True    ## 정규화 확인.
    settings['normalizeScale'] = 1
    settings['removeOutliers'] = 3

    extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
    extractor.enableAllFeatures()

    # Optionally enable some image types(Filters)
    extractor.enableImageTypeByName('Original')
    #extractor.enableImageTypeByName('Wavelet')
    #extractor.enableImageTypeByName('LoG', customArgs={'sigma':[1.0, 2.0, 3.0]})
    #extractor.enableImageTypeByName('Square')
    #extractor.enableImageTypeByName('SquareRoot')
    #extractor.enableImageTypeByName('Exponential')
    #extractor.enableImageTypeByName('Logarithm')
    #extractor.enableImageTypeByName('Gradient')
    #extractor.enableImageTypeByName('LocalBinaryPattern2D')
    #extractor.enableImageTypeByName('LocalBinaryPattern3D')

    # pyradiomics extraction
    featureVector = extractor.execute(image, mask)

    featureVector_df = pd.DataFrame(columns = ['radiomics_feature', 'value'])
    i = 0

    for featureName in featureVector.keys():
        featureVector_df.loc[i] = [featureName, featureVector[featureName]]
        i += 1
    
    # Save the pyradiomics extraction result
    featureVector_df.to_csv(save_path, index=False)
    print(f"Processed: {image_path} with mask {mask_path}, saved in {save_path}")
    save_path.close()
    
def Run_feature_extractor(image_folder, seg_folder, output_folder):
    # 이미지 파일 종류 (1_Pre, 2_AP 등 7가지)
    image_types = ['1_Pre', '2_AP', '3_PP', '4_HP', '5_EQ', '6_T2', '7_HBP']
    # 세그멘테이션 폴더 이름들 (6개)
    seg_folders = ['R1', 'R1_3mm', 'R1_5mm', 'R2', 'R2_3mm', 'R2_5mm']
    # 이미지 폴더에서 고유번호 추출
    unique_ids = set(
        filename.split('_')[-1].replace('.nrrd', '')
        for filename in os.listdir(image_folder)
        if filename.endswith('.nrrd')
    )
    
        # 작업 리스트 생성
    tasks = []
        # 각 고유번호에 대해 처리
    for unique_id in unique_ids:
        # 각 세그멘테이션 폴더에 대해 처리
        for seg_folder_name in seg_folders:
            seg_path = os.path.join(seg_folder, seg_folder_name, f'{unique_id}.nrrd')
            
            # 세그멘테이션 파일 존재 여부 확인
            if not os.path.exists(seg_path):
                print(f"Segmentation file for {unique_id} in {seg_folder_name} not found.")
                continue
            
            # 각 이미지 파일 타입에 대해 처리
            for image_type in image_types:
                image_path = os.path.join(image_folder, f'{image_type}_{unique_id}.nrrd')
                
                # 이미지 파일 존재 여부 확인
                if not os.path.exists(image_path):
                    print(f"Image file {image_type}_{unique_id}.nrrd not found.")
                    continue
                 
                # 출력 경로 설정
                output_dir = os.path.join(output_folder, seg_folder_name)
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(output_dir, f'{image_type}_{unique_id}.csv')
                
                # 이미 처리된 파일인지 확인
                if os.path.exists(output_path):
                    print(f"Output file {output_path} already exists. Skipping.")
                    continue
                
                tasks.append((r"{}".format(image_path), r"{}".format(seg_path), r"{}".format(output_path)))
                # 특징 추출
                #feature_vector = pyradiomics_feature_extractor(image_path, seg_path, output_path)
                #print(image_path, seg_path, output_path)
    
    # 병렬 처리
    #print(tasks)
    NUM_OF_WORKERS = cpu_count() - 1  # 하나의 프로세서는 남겨둠
    if NUM_OF_WORKERS < 1:
        NUM_OF_WORKERS = 1

    with Pool(NUM_OF_WORKERS) as pool:
        pool.map(pyradiomics_feature_extractor, tasks)

'''
if __name__ == '__main__':
    Run_feature_extractor('E:\\2411radiomics\\N4corrected', 'E:\\2411radiomics\\HCCSegmentation', 'E:\\2411radiomics\\bc16_features')
'''


import os
import pandas as pd
from glob import glob

def merge_radiomics_features(input_folder, output_file):
    # 모든 CSV 파일 경로 가져오기
    all_files = glob(os.path.join(input_folder, "*.csv"))

    # 모든 환자의 고유번호와 feature 데이터를 저장할 딕셔너리 초기화
    merged_data = {}

    # 각 파일에 대해 처리
    for file_path in all_files:
        # 파일명에서 고유번호와 시퀀스 정보 추출 (예: 1_Pre_5018644.csv -> sequence: 1_Pre, unique_id: 5018644)
        file_name = os.path.basename(file_path)
        sequence, unique_id = file_name.split('_')[0] + '_' + file_name.split('_')[1], file_name.split('_')[-1].replace('.csv', '')

        # CSV 파일 읽기
        df = pd.read_csv(file_path)

        # unique_id가 이미 존재하지 않으면 새로운 entry 생성
        if unique_id not in merged_data:
            merged_data[unique_id] = {}

        # feature 이름에 sequence 정보 추가하여 데이터 저장
        for _, row in df.iterrows():
            feature_name = f"{sequence}_{row['radiomics_feature']}"
            merged_data[unique_id][feature_name] = row['value']

    # 딕셔너리 데이터를 DataFrame으로 변환
    merged_df = pd.DataFrame.from_dict(merged_data, orient='index')
    merged_df.index.name = 'Patient_ID'  # 고유번호를 A열(Patient_ID)로 설정
    merged_df.reset_index(inplace=True)

    # CSV 파일로 저장
    merged_df.to_csv(output_file, index=False)
    print(f"Data successfully merged into {output_file}")

import pandas as pd

def process_radiomics_data(input_file, output_file):
    # CSV 파일 불러오기
    df = pd.read_csv(input_file)

    # 1. 모든 diagnostics 관련 컬럼 삭제
    diagnostics_columns = [col for col in df.columns if 'diagnostics' in col]
    df.drop(columns=diagnostics_columns, inplace=True)

    # 2. Original_shape 컬럼 처리
    original_shape_columns = [col for col in df.columns if 'original_shape' in col]
    shape_columns_to_keep = [col for col in original_shape_columns if col.startswith('1_Pre_')]
    
    # 1_Pre_로 시작하지 않는 original_shape 컬럼 삭제
    columns_to_drop = [col for col in original_shape_columns if col not in shape_columns_to_keep]
    df.drop(columns=columns_to_drop, inplace=True)
    
    # 남겨진 original_shape 컬럼의 헤더에서 '1_Pre_' 제거
    df.rename(columns={col: col.replace('1_Pre_', '') for col in shape_columns_to_keep}, inplace=True)

    # 3. Kurtosis 컬럼 처리
    kurtosis_columns = [col for col in df.columns if 'Kurtosis' in col]
    
    # Kurtosis 값에서 3을 빼고, 컬럼 이름을 Kurtosis_modified로 변경
    for col in kurtosis_columns:
        df[col] = df[col] - 3
        df.rename(columns={col: col.replace('Kurtosis', 'Kurtosis_modified')}, inplace=True)

    # 4. 추가 컬럼 삭제
    # - original_shape_Maximum2DDiameterSlice, original_shape_Maximum2DDiameterColumn, original_shape_Maximum2DDiameterRow 삭제
    specific_columns_to_delete = [
        'original_shape_Maximum2DDiameterSlice', 
        'original_shape_Maximum2DDiameterColumn', 
        'original_shape_Maximum2DDiameterRow'
    ]
    columns_to_drop = [col for col in df.columns if any(specific in col for specific in specific_columns_to_delete)]
    df.drop(columns=columns_to_drop, inplace=True)
    
    # - 모든 phase에서 original_firstorder_TotalEnergy 및 original_glcm_MCC 삭제
    specific_phase_columns_to_delete = ['original_firstorder_TotalEnergy', 'original_glcm_MCC']
    columns_to_drop = [col for col in df.columns if any(specific in col for specific in specific_phase_columns_to_delete)]
    df.drop(columns=columns_to_drop, inplace=True)
    
    # 5. Patient_ID에 대해 오름차순으로 정렬
    df.sort_values(by='Patient_ID', inplace=True)

    # 결과 CSV 파일로 저장
    df.to_csv(output_file, index=False)
    print(f"Processed data saved to {output_file}")
    

input_folder=r'D:\2411radiomics\Feature_pyradiomics_csv\bc64'
output_folder=r'D:\2411radiomics\IBSI_csv\bc64'

filelist = glob.glob(os.path.join(input_folder, "*.csv"))
for input_path in filelist:
    output_path= os.path.join(output_folder, os.path.basename(input_path))
    os.makedirs(output_folder, exist_ok=True)
    process_radiomics_data(input_path,output_path)


    import pandas as pd
import glob, os

def filter_patients_by_clinical_data(radiomics_file, clinical_file, output_file):
    # radiomics 데이터와 clinical 데이터 로드
    radiomics_df = pd.read_csv(radiomics_file)
    clinical_df = pd.read_csv(clinical_file)

    # clinical 데이터에 있는 Patient_ID 목록 가져오기
    clinical_ids = set(clinical_df['Patient_ID'].unique())

    # radiomics 데이터의 Patient_ID가 clinical 데이터에 있는 경우만 필터링
    filtered_df = radiomics_df[radiomics_df['Patient_ID'].isin(clinical_ids)]
    
    # 제외된 Patient_ID 확인 및 출력
    excluded_ids = set(radiomics_df['Patient_ID'].unique()) - clinical_ids
    if excluded_ids:
        print("Excluded Patient IDs:")
        for pid in sorted(excluded_ids):
            print(pid)
    else:
        print("No Patient IDs were excluded.")

    # 결과를 CSV 파일로 저장
    filtered_df.to_csv(output_file, index=False)
    print(f"Filtered data saved to {output_file}")
    
input_folder=r'C:\Users\KJG\Desktop\radiomics\IBSI_csv\bc128'
output_folder=r'C:\Users\KJG\Desktop\radiomics\IBSI_csv_clinical_exclusion\bc128'
clinical_file_path=r'C:\Users\KJG\Downloads\clinical.csv'

filelist = glob.glob(os.path.join(input_folder, "*.csv"))
for input_path in filelist:
    output_path= os.path.join(output_folder, os.path.basename(input_path))
    os.makedirs(output_folder, exist_ok=True)
    filter_patients_by_clinical_data(input_path, clinical_file_path, output_path)

    
