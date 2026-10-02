# Classifying Neurodegenerative Disease from Quantitative Gait Features
EXSU 500 team project for Group 11 - Dubila Wongibe Trecey Bongayen, Arya Nandhana Syariendrar, Carly Stroll, Anshini Shah

**Research question**

Can machine learning differentiate neurodegenerative disease from healthy controls using quantitative gait features, and which gait characteristics contribute most to classification?

**Dataset**

Gait in Neurodegenerative Disease Database (v1.0.0), PhysioNet https://physionet.org/content/gaitndd/1.0.0/

Created by Hausdorff and colleagues and released openly on PhysioNet. It has gait recordings from 64 adults: 15 with Parkinson's disease, 20 with Huntington's disease, 13 with amyotrophic lateral sclerosis (ALS) and 16 healthy controls. Each participant walked for about five minutes with force-sensitive resistors in their shoes. For each stride, the dataset gives timing measures: stride, swing, stance and double-support intervals for each foot. A separate file lists each participant's age, sex, height, weight and walking speed.

Citation: Hausdorff JM et al. Gait in Neurodegenerative Disease Database. PhysioNet. Goldberger AL et al. PhysioBank, PhysioToolkit, and PhysioNet. Circulation 2000;101(23):e215–e220.

**Problem statement**

Neurodegenerative diseases such as Parkinson’s disease, Huntington’s disease, and amyotrophic lateral sclerosis (ALS) can produce distinct alterations in walking patterns, clinically referred to as gait. This study aims to determine whether machine learning can use quantitative stride-to-stride gait characteristics to classify adults as having Parkinson’s disease, Huntington’s disease, ALS, or being a healthy control. It will also identify which gait characteristics contribute most strongly to distinguishing among these groups.

**Why it matters**

Parkinson's disease, Huntington's disease and ALS all change how people walk, but gait impairment is mostly judged by a clinician's observation, which is subjective and needs specialist time. Wearable insoles can measure stride timing objectively and cheaply. Earlier work found that stride-to-stride variability is markedly higher in these diseases than in healthy adults (Hausdorff et al., 1998). A model that separates patients from controls could support earlier specialist referral. Identifying which gait features matter makes the model's decisions understandable to clinicians, which is essential for trust and adoption in medicine. We will also check whether the model relies on disease-related gait patterns or only on non-specific factors such as slower walking speed and older age.

**Task type**

Binary classification (neurodegenerative disease vs healthy control), with feature-importance analysis.

Known limitations
- Small sample: 64 participants in total, so results are a proof of concept.
- Class imbalance: 48 patients vs 16 controls.
- Pooled diseases: three diseases are combined into one "disease" class, although they affect gait in different ways.
- Confounding: controls and patients may differ in age and walking speed.
