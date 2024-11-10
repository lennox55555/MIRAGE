# MIRAGE 

---

<br>

## Database Structure



**Songs Table:**

| id | artist_name | song_name   |
|----|-------------|-------------|
| 1  | Artist Name | Song Title  |

**Features Table:**

| song_id | stem   | feature_name | feature_values          |
|---------|--------|--------------|-------------------------|
| 1       | vocals | mfccs        | -123.45, -120.67, ...   |
| 1       | vocals | chroma       | 0.12, 0.34, ...         |
| 1       | drums  | mfccs        | -130.56, -125.78, ...   |
| ...     | ...    | ...          | ...                     |

<br>



## Features Dimensionality

MFCCs (20 coefficients):

- The default number of MFCCs in librosa is 20, which captures the most significant spectral characteristics of the audio signal.

Chroma (12 coefficients):

- There are 12 semitones in an octave, hence 12 chroma bins.

Spectral Contrast (7 coefficients):

- By default, librosa computes spectral contrast over 7 frequency bands (6 bands + 1 for the uppermost frequency band).

<br>


## Resources

Main Folder: 

https://drive.google.com/drive/folders/1HCV-JvT3qGOSIKYPMTZ3BZDMC3YK1YfV?usp=drive_link

https://docs.google.com/document/d/13r4rJeDVK2oXUU2blaCPFalAY5RN2bQUMBlERZ3LDJ0/edit?pli=1

XAI Project Thought & Task List:
link to iphone notes app
