# What if HAL Breathed? Enhancing Empathy in Human-AI Interactions with Breathing Speech Synthesis
Main topics: Text-to-Speech Synthesis, Natural Language Processing, Affective Computing, Data Engineering in Python.

Welcome to the repository for the study and thesis titled "What if HAL breathed? Enhancing Empathy in Human-AI Interactions with Breathing Speech Synthesis".

A scientific paper is on the way about this study.

[Read the full thesis here](https://github.com/nicoloddo/PSYCHE/blob/master/Breath_and_speech_synthesis___Master_Thesis.pdf).

## Abstract
Modern Artificial Agents will increasingly leverage AI speech synthesis models to verbally communicate with their users. This study explores the integration of breathing patterns into synthesized speech and their potential to deepen empathy towards said agents, testing the hypothesis that the inclusion of breathing capabilities can significantly enhance the emotional connection in human-AI interaction. 

Breathing patterns have not been unequivocally linked to human emotional states, but respiration has been consistently proven to be involved in emotions' appraisal and regulation, and literature suggests that an inestimable expressive potential may lie behind respiratory noises and their rhythm. Despite this, breathing is hardly involved in speech synthesis models, and literature on breathing agents is still limited. 

We first perform a thorough evaluation of open-source and commercial Speech Synthesis models to understand the breathing synthesis capabilities of state-of-the-art architectures. We then proceed to assess the influence of breathing on the capacity of the voice to evoke empathy. The research methodologically diverges from traditional empathy studies by proposing to the subjects the resolution of an emotional dilemma within a cooperative game scenario, where they face a choice reflecting their empathic engagement with an AI partner.

The findings indicate that breathing in synthesized speech significantly enhances agents' perceived naturalness and users' empathy towards them. These insights underscore the importance of breathing in speech synthesis for AI design and call for its consideration in future models and interactive Artificial Agents. Ultimately, the study aims to contribute to the development of a more empathetic digital world through enhanced human-AI interaction.

Interested in experiencing the gamified dilemma first-hand? [Click here](https://nicoloddo.github.io/Psyche) to try it out.

Listen to the synthesized speech incorporating breathing patterns at [PsycheRecordings](https://nicoloddo.github.io/PsycheRecordings/).

The comprehensive dataset of the study results are available on Kaggle. Find the dataset [here](https://www.kaggle.com/datasets/nicoloddo/psyche-empathy-dataset) and the results analysis in this [Kaggle notebook](https://www.kaggle.com/code/nicoloddo/what-if-hal-breathed-results-analysis).

## This repository contains materials used for the study and the preprocessing tool. Additional repositories relevant to the study are listed below.

### Speech Synthesis Deep Learning Models
- We analyzed the state-of-the-art speech synthesis models to understand the possibilities of integrating breathing and spontaneous speech patterns. A broad list of open-source and commercial synthesizers that we tested is present at the full thesis link.
- We attempted to train two open-source models: VITS and Flowtron, with some modifications to their architecture, which can be found at these repos: [psyche-vits](https://github.com/nicoloddo/pysche-vits), [psyche-flowtron](https://github.com/nicoloddo/flowtron). However, due to limitations in computational resources, we finally adopted the pre-trained model [BARK](https://github.com/suno-ai/bark).
- We found BARK to be the only model suitable to be applied in our study, highlighting a lack of models that can achieve speech-breathing synthesis.
- We applied iterative prompt engineering techniques to the BARK model to synthesize spontaneous speech with emotional features and integrated breathing patterns.

### Data Processing
- Various Speech-to-Text services and cloud computing platforms were employed for transcription of audio databases, including IBM Cloud, AssemblyAI, and Google Cloud.
- A Preprocessing and Breath Labeling tool was developed to process speech databases. This software is part of this repository at the folder Preprocessingi/ but is still undergoing refinement for broader usability.

### Gamified Experiment
- We designed a gamified experience using Unity Engine and C# to evaluate the emotional impact of breathing in synthesized voices for virtual agents. The game-based approach offered a unique angle for this assessment. The gamified experience's code is hosted on [PSYCHE-Gamified](https://github.com/nicoloddo/PSYCHE-Gamified).