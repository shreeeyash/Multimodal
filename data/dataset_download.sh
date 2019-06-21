makedir dataset
cd dataset

wget "http://immortal.multicomp.cs.cmu.edu/raw_datasets/old_processed_data/cmu-mosei/data/audio_test.h5"
wget "http://immortal.multicomp.cs.cmu.edu/raw_datasets/old_processed_data/cmu-mosei/data/audio_train.h5"
wget "http://immortal.multicomp.cs.cmu.edu/raw_datasets/old_processed_data/cmu-mosei/data/text_test_emb.h5"
wget "http://immortal.multicomp.cs.cmu.edu/raw_datasets/old_processed_data/cmu-mosei/data/text_train_emb.h5"
wget "http://immortal.multicomp.cs.cmu.edu/raw_datasets/old_processed_data/cmu-mosei/data/video_test.h5"
wget "http://immortal.multicomp.cs.cmu.edu/raw_datasets/old_processed_data/cmu-mosei/data/video_train.h5"

wget "http://immortal.multicomp.cs.cmu.edu/raw_datasets/old_processed_data/cmu-mosei/labels/mosi2uni_Test_labels_angry.csv"
wget "http://immortal.multicomp.cs.cmu.edu/raw_datasets/old_processed_data/cmu-mosei/labels/mosi2uni_Test_labels_disgust.csv"
wget "http://immortal.multicomp.cs.cmu.edu/raw_datasets/old_processed_data/cmu-mosei/labels/mosi2uni_Test_labels_fear.csv"
wget "http://immortal.multicomp.cs.cmu.edu/raw_datasets/old_processed_data/cmu-mosei/labels/mosi2uni_Test_labels_happy.csv"
wget "http://immortal.multicomp.cs.cmu.edu/raw_datasets/old_processed_data/cmu-mosei/labels/mosi2uni_Test_labels_sad.csv"
wget "http://immortal.multicomp.cs.cmu.edu/raw_datasets/old_processed_data/cmu-mosei/labels/mosi2uni_Test_labels_surprise.csv"

wget "http://immortal.multicomp.cs.cmu.edu/raw_datasets/old_processed_data/cmu-mosei/labels/mosi2uni_Train_labels_angry.csv"
wget "http://immortal.multicomp.cs.cmu.edu/raw_datasets/old_processed_data/cmu-mosei/labels/mosi2uni_Train_labels_disgust.csv"
wget "http://immortal.multicomp.cs.cmu.edu/raw_datasets/old_processed_data/cmu-mosei/labels/mosi2uni_Train_labels_fear.csv"
wget "http://immortal.multicomp.cs.cmu.edu/raw_datasets/old_processed_data/cmu-mosei/labels/mosi2uni_Train_labels_happy.csv"
wget "http://immortal.multicomp.cs.cmu.edu/raw_datasets/old_processed_data/cmu-mosei/labels/mosi2uni_Train_labels_sad.csv"
wget "http://immortal.multicomp.cs.cmu.edu/raw_datasets/old_processed_data/cmu-mosei/labels/mosi2uni_Train_labels_surprise.csv"

cd ..