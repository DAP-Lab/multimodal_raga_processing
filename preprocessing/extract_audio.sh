#!/bin/sh

SOURCE_DIRECTORY='./PuneRecording'
TARGET_DIRECTORY='./PuneRecordingAudio'
RENAMED_AUDIO_DIRECTORY='./PuneRecordingRenamed'

ROOT_DIRECTORY='/media/antpc/NewDisk/research_work/'
SPLIT_AUDIO_DIRECTORY='./audio_split'
AUDIO_START_END_DIRECTORY='./start_and_stop_times/withStartAndStopTimes'
RAW_AUDIO_CROPPED='./raw_audio_cropped'
VOCALS_CROPPED='./vocals_cropped'
TONIC_FOLDER='./SingerSpecificTonic'

for each_dir in `ls $SOURCE_DIRECTORY`
do
    if [ ! -d $TARGET_DIRECTORY/$each_dir ]
    then
        mkdir $TARGET_DIRECTORY/$each_dir
    fi
    for each_subdir in `ls $SOURCE_DIRECTORY/$each_dir`
    do
        if [ ! -d $TARGET_DIRECTORY/$each_dir/$each_subdir ]
        then
            mkdir $TARGET_DIRECTORY/$each_dir/$each_subdir
        fi
    done
done

for each_dir in `ls $SOURCE_DIRECTORY`
do
    for each_subdir in `ls $SOURCE_DIRECTORY/$each_dir`
    do
        list_of_video_files=`ls $SOURCE_DIRECTORY/$each_dir/$each_subdir`
        for each_file in `echo $list_of_video_files`
        do
            source_file_name=`echo $SOURCE_DIRECTORY/$each_dir/$each_subdir/$each_file`
            output_file_name=`echo $TARGET_DIRECTORY/$each_dir/$each_subdir/$each_file | sed -e 's/.mp4/.aac/'`
            ffmpeg -i $source_file_name -vn -acodec copy $output_file_name          
        done
    done
done

for each_dir in `ls $TARGET_DIRECTORY`
do
    for each_subdir in `ls $TARGET_DIRECTORY/$each_dir`
    do
        list_of_audio_files=`ls $TARGET_DIRECTORY/$each_dir/$each_subdir`
        for each_file in `echo $list_of_audio_files`
        do
            source_file_name=`echo $TARGET_DIRECTORY/$each_dir/$each_subdir/$each_file`
            output_file_name=`echo $TARGET_DIRECTORY/$each_dir/$each_subdir/$each_file | sed -e 's/.aac/.wav/'`
            ffmpeg -i $source_file_name $output_file_name
            rm $source_file_name        
        done
    done
done

for each_dir in `ls $TARGET_DIRECTORY`
do
    for each_subdir in `ls $TARGET_DIRECTORY/$each_dir`
    do
        list_of_audio_files=`ls $TARGET_DIRECTORY/$each_dir/$each_subdir | grep -i Front`
        for each_file in `echo $list_of_audio_files/*`
        do
            echo "Processing $each_file"
            each_file_full_path=$TARGET_DIRECTORY/$each_dir/$each_subdir/$each_file
            spleeter separate -p spleeter:4stems -o audio_split $each_file_full_path
        done
    done
done


for each_dir in `ls $RAW_AUDIO_CROPPED`
do
    for each_subdir in `ls $TARGET_DIRECTORY/$each_dir`
    do
        list_of_audio_files=`ls $RAW_AUDIO_CROPPED | sort`
        for each_file in `echo $list_of_audio_files`
        do
            echo "Processing $each_file"
            each_file_full_path=$RAW_AUDIO_CROPPED/$each_file
            spleeter separate -p spleeter:4stems -o audio_split_4stems $each_file_full_path
        done
    done
done

for each_dir in `ls $SPLIT_AUDIO_DIRECTORY`
do
    echo "**********************"
    output_file_name=$each_dir.wav
    mv $SPLIT_AUDIO_DIRECTORY/$each_dir/vocals.wav $output_file_name
    mv  $SPLIT_AUDIO_DIRECTORY/$each_dir/$output_file_name $SPLIT_AUDIO_DIRECTORY/$output_file_name
    echo "rm $SPLIT_AUDIO_DIRECTORY/$each_dir/bass.wav $SPLIT_AUDIO_DIRECTORY/$each_dir/other.wav $SPLIT_AUDIO_DIRECTORY/$each_dir/drums.wav"
    echo "mv $SPLIT_AUDIO_DIRECTORY/$each_dir/$output_file_name ../"
    echo "rmdir $SPLIT_AUDIO_DIRECTORY/$each_dir"
done

for each_file in `ls $AUDIO_START_END_DIRECTORY/*.wav`
do
    echo "******************************************************************"
    corresponding_txt_file=`echo $each_file | sed -e 's/wav/txt/g'`
    start=`cat $corresponding_txt_file | awk '{print $1}'`
    end=`cat $corresponding_txt_file | awk '{print $2}'`
    #
    each_file_basename=`basename $each_file`
    front_camera_file_name=`echo $each_file_basename | sed -e 's/\([A-Z][A-Z]\)_\(.*\)/\1_Front_Camera_\2/g'`
    original_wav_file=$RENAMED_AUDIO_DIRECTORY/$front_camera_file_name
    cropped_original_wav_file=$RAW_AUDIO_CROPPED/$each_file_basename
    #
    wav_vocals=$SPLIT_AUDIO_DIRECTORY/$each_file_basename
    cropped_vocal_file=$VOCALS_CROPPED/$each_file_basename  
    #
    ffmpeg -i $original_wav_file -ss $start -to $end -acodec copy $cropped_original_wav_file
    ffmpeg -i $wav_vocals -ss $start -to $end -acodec copy $cropped_vocal_file
done

for each_file in `ls $RAW_AUDIO_CROPPED`
do
    inputfile=$RAW_AUDIO_CROPPED/$each_file
    outputfilename=`basename $each_file | sed -e 's/wav/txt/g'`
    outputfilename_full_path=$TONIC_FOLDER/$outputfilename
    python extract_tonic.py $inputfile > $outputfilename_full_path
done

for each_file in `ls $VOCALS_CROPPED`
do
    echo "Running for $each_file"
    inputfile=$VOCALS_CROPPED/$each_file
    singer_name=`basename $each_file | cut -c1-2`
    tonic_file_name=${singer_name}_tonic.txt
    tonicfilename_full_path=$TONIC_FOLDER/$tonic_file_name
    python extract_pitch_contours.py $inputfile  $tonicfilename_full_path
done

# for each_file in `ls $VOCALS_CROPPED`
# do
#     echo "Running for $each_file"
#     inputfile=$VOCALS_CROPPED/$each_file
#     singer_name=`basename $each_file | cut -c1-2`
#     tonic_file_name=${singer_name}_tonic.txt
#     tonicfilename_full_path=$TONIC_FOLDER/$tonic_file_name
#     python extract_pitch_contours.py $inputfile
# done