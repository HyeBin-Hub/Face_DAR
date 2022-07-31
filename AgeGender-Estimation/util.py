import wget

# model : https://gist.github.com/GilLevi/c9e99062283c719c03de#file-deploy_age-prototxt
# prototxt : https://github.com/spmallick/learnopencv/tree/master/AgeGender

face_caffe_model_url = "https://github.com/spmallick/learnopencv/raw/master/FaceDetectionComparison/models/res10_300x300_ssd_iter_140000_fp16.caffemodel"
face_pototxt_url = "https://raw.githubusercontent.com/spmallick/learnopencv/master/FaceDetectionComparison/models/deploy.prototxt"

age_caffe_model_url = "https://github.com/GilLevi/AgeGenderDeepLearning/raw/master/models/age_net.caffemodel"
gender_caffe_model_url = "https://github.com/GilLevi/AgeGenderDeepLearning/raw/master/models/gender_net.caffemodel"

age_pototxt_url = "https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/age_deploy.prototxt"
gender_pototxt_url = "https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/gender_deploy.prototxt"

output_path = "./caffe_model"

# model download
wget.download(age_caffe_model_url, output_path)
wget.download(gender_caffe_model_url, output_path)
wget.download(face_caffe_model_url, output_path)


# model prototxt download
wget.download(age_pototxt_url, output_path)
wget.download(gender_pototxt_url, output_path)
wget.download(face_pototxt_url, output_path)