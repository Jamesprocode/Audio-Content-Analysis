#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This activity is designed to be pseudo code, not code (even if you think
you know what you are doing.) Pseudo code is writing down steps *like* an 
algorithm but without worrying about exact functions or syntax.
You'll use plain English mixed with code-y words or steps (like if, while, for,
etc.). 

The following is an example of an everyday process broken into pseudo code
(sorting and washing laundry):
   
#######################################
Load laundry basket with dirty clothes
FOR each item in laundry basket:
    Make piles:
    IF item is white:
        put item in white_pile
        
    ELSE IF item is dark:
        put item in delicate_pile       
        
For Each_pile:
    put pile in washing machine
    add detergent
    start washer
    wait until done
    move pile to dryer
    wait until done
#######################################  

Write up in pseudo code ONLY the steps you would need to take an audio input 
(file or filepath), and return the audio broken up into short, potentially 
overlapping frames. These frames will be in a 2D vector or array with each 
frame as a single array of audio nested inside the main array, and you 
will also need to return the timestamp location for the beginning of each new
frame.
    
You will think through:
    * What input will you have?
    * What will be the final output?
    * What parameters need to be defined?
    * What can be set as default and what needs user input?
    * What are the steps involved?
    * How do you loop through the audio?
"""

    
'''
load audio file from filepath or audio input

define blocking size for frames (e.g., 1024 samples)
define overlap size for frames (e.g., 512 samples)

initialize empty 2D array or vector for audio frames


while there are more samples in audio:
    calculate current position in audio
    calculate start index for frame based on current position
    calculate end index for frame based on start index and blocking size and overlap size
    IF end index exceeds total audio length:
        adjust end index to total audio length
    
    extract audio segment from start index to end index
    store audio segment in 2D array or vector

    update current position for next frame

return 2D array of audio frames

'''

# Pseudo code for breaking audio into frames
# load audio file from filepath or audio input
# define blocking size for frames (e.g., 1024 samples)
# define overlap size for frames (e.g., 512 samples)

# initialize empty 2D array or vector for audio frames

# while there are more samples in audio:
#     calculate current position in audio
#     calculate start index for frame based on current position
#     calculate end index for frame based on start index and blocking size and overlap size

#     IF end index exceeds total audio length:
#         adjust end index to total audio length

#     extract audio segment from start index to end index
#     store audio segment in 2D array or vector

#     update current position for next frame

# return 2D array of audio frames

# '''

# # Pseudo code for breaking audio into framesd