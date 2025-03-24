# OpenLabCluster

## Usage
### Installation
#### Install from pip
Create a new environment with conda and install the package from environment.yml file
conda env create -f environment.yml


For **Linux**, **Mac OS** and **Windows** create environment (required conda version > 24.9.2)

   	conda env create -f environment.yml

For windows users, following packages might need to be installed manually

      pip install streamlit==1.26.0
      pip install umap-learn
      pip install streamlit-plotly-events
      pip install opencv-python
      pip install st_pages==v0.4.1
      conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 cpuonly -c pytorch

	
### Execution

Run the following for **Linux** and **Mac-OS**
	
	streamlit run main_page.py

Open the link in your browser (Chrome is preferred).

Make sure you follow the steps: (1) Project Manager -> (2) Cluster Map -> (3) Behavior Classification Map. 

		
### Run a Demo

#### Create Demo Project
1. Download the *openlabcluster_example* folder from [here](https://drive.google.com/file/d/1N-NMLGSkYTiS1lIRg8jwEZjoORDx_W6p/view?usp=sharing)
1. Go to *your_download-dir/openlabcluster_example* folder run 
		
		python3 prepare_video_list.py
   which generates video_segmetns_names.text file.
   
2. Launch OpenLabCluster GUI (see Execution above)
3. Go to **Project Manager**
4. Chose the project loading method: Create New Project 
5. Set Project Name: e.g., demo 
6. Enter the directory to keypoints or precomputed kinematics, e.g., your_download_dir/openlabcluster_example (/home/ws2/OLC_Project/openlabcluster_example)
7. Enter the filename of keypoints or precomputed kinematics (h5 file) , e.g., demo.h5 
8. Click *Load Video Segments Name List*, choose the file: your_download_dir/openlabcluster_example/video_segmetns_names.text 
9. Uncheck *Check to use GPU*, if GPU is not available. 
10. Set Feature Length = 16 
11. Go to **Cluster Map**

#### Start the Demo Project:
1. Go to **Manage Project** panel, 
2. Chose the project loading method: Load Project
3. Select the config file as */your_home_dir/OLC_Project/project_name/config.yaml*
4. Go to **Cluster Map**


#### Cluster Map:
1. Click **Start Clustering** button to start unsupervised clustering. 
2. Click **Go To Classification** when unsupervised clustering is finished, then go to **Behavior Classification Map** panel.

#### Behavior Classification Map: 
1. The scatter plot indicating sample clusters is initialized on the bottom left, with suggested samples for annotation.
2. Label samples on the bottom right panel.
3. Click **Run Classification**, and start classification.


### Manage Project (Start a New Project or Load Earlier Project) Detailed Description
#### Start a New Project
1. Project Name - the name of the project
2. If you have only videos, use markerless pose estimators (e.g., DeepLabCut) to extract keypoints for each video first. If you already have DeepLabCut-like formatted files, enter the directory of your dataset.
3. Enter the filename of keypoints or precomputed kinematics.
4. List video names in the "Load Video Segments Names List" file. There should be one video for each keypoint file. Make sure the videos and keypoints files are in the same order.
5. Keep the GPU box checked if you have a GPU on your computer, and you would like to use it for training.
6. Enter the feature length (number of body parts * number of dimensions per body part. For example, 5 keypoints in 2D would be 5*2 =10).

#### Loading a Project
1. Select the config.yaml file generated when you created that project.
2. Press "OK."


### Cluster Map (Unsupervised Learning)
#### Set the Training Parameters
1. Update Cluster Map Every (Epochs): The frequency to update the Cluster Map (bottom left panel in the figure), e.g., set 1 to update Cluster Map every training epoch, or 5 to update every 5 epochs.
2. Save Cluster Map Every (Epochs): The frequency to save the Cluster Map.
3. Maximum Epochs: The maximum number of epochs to be performed unsupervised training.
4. Cluster Map Dimension: Possible choices are 2d or 3d. For 2d, the Cluster Map will be shown in two dimensions; otherwise, it will be three dimensions.
5. Dimension Reduction Methods: Possible choices are PCA, tSNE, and UMAP. The GUI will use the chosen method to perform dimension reduction and show results in Cluster Map.


#### Buttons
After setting the parameters, you can perform the analysis:

1. Start Clustering: Perform an unsupervised sequence regeneration task.
2. Stop Clustering: Usually, the clustering will stop when it reaches the maximum epochs, but if you want to stop at an intermediate stage, click this button.
3. Continue Clustering: If you stopped the clustering at some stage and want to perform clustering with earlier clustering results, click this button.
4. Go to Classification: After the unsupervised clustering, we go to the next step, which includes: i) annotation suggestion, ii) sample annotating, and iii) semi-supervised action classification with labeled samples.

### Behavior Classification Map
#### Set the Training Parameters
1. Selection Method: In this part, your selection will decide which method GUI uses to select samples for annotation. There are four possible choices: Marginal Index (MI), Core Set (CS), Cluster Center (TOP), Cluster Random (Rand), and Uniform.
2. \# Samples per Selection: the number of samples you want to label in the current selection stage. You can select or deselect samples in the behavior classification map.
3. Maximum Epochs: The maximum epoch the network will be trained when performing the action recognition.
4. Cluster Map Dimension: You can choose 2d or 3d. If it is 2d, the Cluster Map will be shown in two-dimensional space; otherwise, it is three-dimensional space.
5. Dimension Reduction Method: Possible choices are PCA, tSNE, and UMAP. The GUI will use the chosen reduction method to perform dimension reduction and show results in the Cluster Map.


#### Buttons:
   
1. Run Classification: Save annotation results and train the action recognition model.
   
2. Stop Classification: Stop training.
   
3. Next Selection: Suggest a new set of samples based on with indicated active learning method. **Notice**: If you change Cluster Map Dimension or Dimension Reduction Method, click the **Next Selection** to show suggested samples.

4. Get Results: Get the Behavior Classification Map (predicted class label) from the trained model on unlabeled samples. 

#### Plots
1. Behavior Classification Plot:
   
      Dots for each action segment in a different color are shown in the Behavior Classification plot (only in 2D).
      Red: current sample for annotating, and the video segment is shown on the right. 
      Blue: the suggested samples for annotating in this iteration (deselect them by clicking the point).
      Green: samples have been annotated.
      Gray: reamining samples

   Buttons:
   
     * Zoom: Zoom in or zoom out the plot.
     * Pan: Move the plot around.
   
   
2. Videos Panel:
   
   Left panel: The corresponding video of the action segment, which is shown in the Behavior Classification Map in red.
   
   Right panel: The class name and class id. According to the video, one can label the action segment.

   Buttons:

     * Previous: Load the previous video.
     * Play: Play the video.
     * Next: Go to the next video.


