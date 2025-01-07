# Player Statistics Predictor

Player Statistics Predictor is a Python-based project designed to predict season statistics for players in an online college basketball game (http://onlinecollegebasketball.org). By leveraging web scraping, data preprocessing, and machine learning techniques, this project aims to provide accurate predictions of player performance metrics.

## Features

- **Web Scraping**: Utilized BeautifulSoup to collect thousands of player statistics from an online college basketball game and store them in a CSV file.
- **Data Preprocessing**: Cleaned and processed over 10,000 data points to ensure the machine learning model received high-quality input data.
- **Machine Learning**: Implemented a Ridge Regression model to predict player statistics, enhancing predictive accuracy through advanced preprocessing techniques.
- **Comprehensive Pipeline**: Integrated data collection, preprocessing, model training, and prediction into a streamlined process.

## Technologies Used

- **Programming Language**: Python
- **Libraries and Tools**:
  - BeautifulSoup (Web scraping)
  - Scikit-Learn (Machine learning)
  - Pandas (Data manipulation and analysis)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/brherold/Player-Statistics-Predictor.git
   ```
2. Navigate to the project directory:
   ```bash
   cd Player-Statistics-Predictor
   ```
3. Install the required Python libraries:
   ```bash
   pip install beautifulsoup4 scikit-learn pandas
   ```

## Usage
1. **Make predictions** for player stats:
   ```bash
   python getPredictedStats.py
   ```


