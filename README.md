<h2>Social Media Trend Prediction and Recommender Systems</h2>

Social media used is Youtube API

Description
This project analyzes trending YouTube videos and provides recommendations based on a user-specified topic. It uses YouTube’s Data API to fetch video data such as views, likes, comments, and publish time, and applies machine learning (Random Forest) to classify videos as "Trending" or "Non-Trending." It also visualizes various metrics like trending video distribution and engagement metrics.

Features
Fetches Trending Videos from YouTube based on view count, like count, and comment count.

Classifies videos as "Trending" or "Non-Trending" based on engagement metrics.

Visualizes trending video distribution and hourly/daily patterns.

Recommends top videos by topic based on user input.

Visualizes engagement metrics (views, likes, comments) for top recommended videos.

Requirements
The project uses the following Python libraries:

matplotlib - for data visualization

requests - for making HTTP requests to YouTube API

pandas - for handling and analyzing data

seaborn - for statistical data visualization

scikit-learn - for training the Random Forest classifier

To install the required libraries, create a virtual environment and use the following command to install all dependencies:


pip install -r requirements.txt

Setup

Create a YouTube Data API Key:

1. Go to the Google Cloud Console.

2. Create a new project.

3. Enable the YouTube Data API v3 under API & Services > Library.

4. Create an API Key in the Credentials section.

5. Update API Key in the Code: In the backend.py file, replace the placeholder API_KEY with your actual API Key.

API_KEY = "YOUR_API_KEY_HERE"

Run the Project:

Once you have set up the environment and API Key, run the script:

python backend.py

Interacting with the Program:

The program will fetch trending YouTube videos and classify them.

It will then ask you to input a topic to fetch and recommend top videos related to that topic.

Visualizations will be displayed to help you understand video trends and engagement.

Usage
1. Fetching Trending Videos:
The script automatically fetches trending videos in the U.S. (can be modified for other regions).

2. Classifying Videos:
The videos are classified into two categories: Trending or Non-Trending, based on their view count and like count.

3. Recommendation System:
The user is prompted to enter a topic (e.g., "technology", "sports", etc.), and the top 10 related videos are recommended.

4. Visualizations:

A distribution plot shows the number of trending vs non-trending videos.

A heatmap visualizes the trendingness of videos based on their publish time (hour of day vs day of week).

A bar plot compares the engagement metrics (views, likes, comments) for the top 10 recommended videos.

Outputs:
![metrics](https://github.com/user-attachments/assets/740265a5-5bb9-472a-83ff-cd17a5ebfe2d)
![trending_vs_nontrending](https://github.com/user-attachments/assets/31854a2c-ec7e-48e4-8aac-f7341e1d2daf)
![heatmap](https://github.com/user-attachments/assets/b2065871-0bbf-4079-904e-140dc1bf3187)
![recommendation](https://github.com/user-attachments/assets/3e7c295f-ef3d-405c-a2f3-ecabe91417bd)
![engagement](https://github.com/user-attachments/assets/1537f5b2-0ee4-4c0f-938c-6fcc99e86fd2)


Troubleshooting

1. API Key Error:

If you encounter an error regarding the API key, double-check the key’s validity and permissions.

2. Rate Limit Issues:

YouTube’s API has rate limits. If you exceed them, the script will fail. You can increase your quota via the Google Cloud Console.


Acknowledgements
This project uses the YouTube Data API v3 provided by Google.

The Random Forest classifier is implemented using the scikit-learn library.

Contact
For any questions or issues, feel free to open an issue in the repository or contact Shanmukhi Poluri at shanmukhip2005@gmail.com.

