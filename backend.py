import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ---- Global Variables ----
API_KEY = "your_api_key"
TRENDING_API_URL = "https://www.googleapis.com/youtube/v3/videos"
SEARCH_API_URL = "https://www.googleapis.com/youtube/v3/search"
VIDEO_DETAILS_URL = "https://www.googleapis.com/youtube/v3/videos"
MAX_RESULTS = 100
CSV_FILE = "youtube_trending_data.csv"


# ---- Step 1: Fetch Trending Videos ----
def fetch_trending_videos(api_key, max_results=100):
    params = {
        "part": "snippet,statistics",
        "chart": "mostPopular",
        "regionCode": "US",
        "maxResults": min(max_results, 50),
        "key": api_key
    }

    all_items = []
    nextPageToken = None

    while len(all_items) < max_results:
        if nextPageToken:
            params['pageToken'] = nextPageToken

        response = requests.get(TRENDING_API_URL, params=params)
        data = response.json()

        items = data.get("items", [])
        all_items.extend(items)

        nextPageToken = data.get("nextPageToken")
        if not nextPageToken:
            break

    videos = []
    for item in all_items[:max_results]:
        snippet = item["snippet"]
        statistics = item["statistics"]

        videos.append({
            "video_id": item["id"],
            "title": snippet.get("title"),
            "publishedAt": snippet.get("publishedAt"),
            "channelTitle": snippet.get("channelTitle"),
            "viewCount": int(statistics.get("viewCount", 0)),
            "likeCount": int(statistics.get("likeCount", 0)),
            "commentCount": int(statistics.get("commentCount", 0)),
            "categoryId": snippet.get("categoryId", "0")
        })

    return pd.DataFrame(videos)


# ---- Step 2: Save Data to CSV ----
def save_to_csv(df, filename):
    df.to_csv(filename, index=False)
    print(f"Data saved to {filename}")


# ---- Step 3: Prepare Data ----
def prepare_data(df):
    df['publishedAt'] = pd.to_datetime(df['publishedAt'])
    df['publishedHour'] = df['publishedAt'].dt.hour
    df['publishedDay'] = df['publishedAt'].dt.dayofweek

    df['isTrending'] = ((df['viewCount'] > df['viewCount'].median()) &
                        (df['likeCount'] > df['likeCount'].median())).astype(int)

    features = df[['viewCount', 'likeCount', 'commentCount', 'publishedHour', 'publishedDay']]
    target = df['isTrending']

    return features, target, df


# ---- Step 4: Train Random Forest ----
def train_random_forest(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Accuracy Score:", accuracy_score(y_test, y_pred))

    return clf, X_test, y_test, y_pred


# ---- Step 5: Visualizations for Trending ----
def plot_trending_distribution(df):
    sns.countplot(x='isTrending', data=df)
    plt.title("Number of Trending vs Non-Trending Videos")
    plt.xlabel("Is Trending")
    plt.ylabel("Count")
    plt.xticks([0, 1], ['Non-Trending', 'Trending'])
    plt.show()


def plot_heatmap(df):
    pivot = df.pivot_table(values='isTrending', index='publishedHour', columns='publishedDay', aggfunc='mean')
    sns.heatmap(pivot, cmap="YlGnBu", annot=True, fmt=".2f")
    plt.title("Trendingness Heatmap (Hour vs Day)")
    plt.xlabel("Day of Week (0=Mon)")
    plt.ylabel("Hour of Day")
    plt.show()


# ---- Step 6: Fetch Top 10 Videos by Topic ----
def fetch_top_videos_by_topic(api_key, topic, max_results=10):
    search_params = {
        "part": "snippet",
        "q": topic,
        "type": "video",
        "maxResults": max_results,
        "key": api_key
    }

    search_response = requests.get(SEARCH_API_URL, params=search_params)
    search_data = search_response.json()

    video_ids = [item['id']['videoId'] for item in search_data['items']]

    details_params = {
        "part": "snippet,statistics",
        "id": ",".join(video_ids),
        "key": api_key
    }

    details_response = requests.get(VIDEO_DETAILS_URL, params=details_params)
    details_data = details_response.json()

    videos = []
    for item in details_data['items']:
        snippet = item['snippet']
        stats = item['statistics']
        videos.append({
            "video_id": item["id"],
            "title": snippet.get("title"),
            "channelTitle": snippet.get("channelTitle"),
            "viewCount": int(stats.get("viewCount", 0)),
            "likeCount": int(stats.get("likeCount", 0)),
            "commentCount": int(stats.get("commentCount", 0))
        })

    return pd.DataFrame(videos)


# ---- Step 7: Visualize Engagement Metrics ----
def plot_engagement_metrics(df):
    df_sorted = df.sort_values(by="viewCount", ascending=False)

    plt.figure(figsize=(12, 6))
    bar_width = 0.25
    indices = range(len(df_sorted))

    plt.bar(indices, df_sorted['viewCount'], width=bar_width, label='Views')
    plt.bar([i + bar_width for i in indices], df_sorted['likeCount'], width=bar_width, label='Likes')
    plt.bar([i + 2 * bar_width for i in indices], df_sorted['commentCount'], width=bar_width, label='Comments')

    plt.xlabel('Video Titles', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title('Top 10 Videos Engagement Metrics', fontsize=14)
    plt.xticks([i + bar_width for i in indices], df_sorted['title'], rotation=90, ha='center', fontsize=8)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ---- MAIN FUNCTION ----
def main():
    print("\n=== Fetching Trending Videos ===")
    trending_df = fetch_trending_videos(API_KEY, MAX_RESULTS)
    save_to_csv(trending_df, CSV_FILE)

    print("\n=== Preparing Data ===")
    X, y, processed_df = prepare_data(trending_df)

    print("\n=== Training Random Forest Classifier ===")
    model, X_test, y_test, y_pred = train_random_forest(X, y)

    print("\n=== Visualizing Trending Video Analysis ===")
    plot_trending_distribution(processed_df)
    plot_heatmap(processed_df)

    print("\n=== Recommending Top 10 Videos by Topic ===")
    topic = input("Enter a topic to recommend top videos: ")
    top_videos_df = fetch_top_videos_by_topic(API_KEY, topic, max_results=10)
    print(top_videos_df[['title', 'viewCount', 'likeCount', 'commentCount']])

    print("\n=== Visualizing Engagement Metrics ===")
    plot_engagement_metrics(top_videos_df)


# ---- RUN ----
if __name__ == "__main__":
    main()
