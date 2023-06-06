# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 21:29:08 2023

@author: nicol
"""
import argparse
import transcript_align_lib as tal
import json
import nic_dataset_tools as ndt

tal.dataset_dir_relative = "INTERSPEECH/ComParE2020_Breathing/"
tal.breath_labels_dir_relative = 'lab/labels.csv'
tal.transcriptions_dir_relative = 'transcriptions_and_alignments/'
tal.wavs_dir_relative = "normalized_wav/"
tal.temp_wavs_dir_relative = "temp_wav/"

def main(args):
    
    # Trancript with AssemblyAI
    transcript_ids = tal.assai_queue_transcripts(args.assai_apikey)
    tal.assai_get_transcripts(transcript_ids, args.assai_apikey)
    
    print("Transcriptions done. Now use Montreal Forced Aligner (which is yet not supported to be started from here), and start the Gentle Aligner Docker image.")
    input("Press Enter to continue.")
    
    # Align with Gentle (also align with MFA separately!)
    # gentle_align(relative directory in which the transcripts are saved, directory where to save the alignment, directory where to save the sorted alignment)
    tal.gentle_align('transcript_assemblyAI/', 'aligned_assemblyAI/', 'sorted_aligned_assemblyAI/', args.gentle_docker_port)
    
    # gentle_mfa_align(directory of the Gentle non-sorted alignment, directory of the MFA alignment, directory to save the merge alignment)
    tal.gentle_mfa_align('aligned_assemblyAI/json/', 'MFAFormatCorpus/aligned_assai/', 'gentle_mfa_assemblyAI/')
    
    print("Done!")
    print("Now you can use label_breath.py to label the breathing in the transcriptions and segment_by_breath to segment the audio cutting on the breathing instances.")
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--assai_apikey', type = str, default = ndt.ASSAI_APIKEY,
        help = '')
    
    parser.add_argument('--gentle_docker_port', type = str, default = '32768',
        help = '')

    args = parser.parse_args()
    main(args)