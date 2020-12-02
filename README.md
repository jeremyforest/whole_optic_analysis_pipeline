OPTIMAS: whOle-oPTical Imaging AnalysiS pipeline

HIgh throughput automated analysis pipeline to pre-process and analyse whole optical electrophysiology data. 
Under development. Subject to major chamges.

**To install with anaconda**
~~~
conda create -n OPTIMAS
conda activate OPTIMAS
git clone git@github.com:jeremyforest/whole_optic_gui.git
pip install -e .
~~~


**To run:**
~~~
conda activate OPTIMAS
python OPTIMAS/pipeline.py --input_data_folder
~~~


**Data needs to be organized in the following way:**

Date                                      <---- input folder (enter absolute path)    \
  |--- Experiment A                                                                   \
&nbsp;&nbsp;|--- Raw Data                                                             \
&nbsp;&nbsp;|--- experiment_A_timings.json                                             \
&nbsp;&nbsp;|--- experiment_A_info.json                                                \
  |--- Experiment B                                                                   \
&nbsp;&nbsp;|--- Raw Data                                                             \
&nbsp;&nbsp;|--- experiment_B_timings.json                                             \
&nbsp;&nbsp;|--- experiment_B_info.json                                                \



