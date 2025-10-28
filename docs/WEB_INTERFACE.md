# F1 Race Predictor - Web Interface User Guide

Welcome to the F1 Race Predictor web application! This guide will help you navigate and use all the features available in our Streamlit-based interface.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Navigation](#navigation)
3. [Features](#features)
4. [Settings](#settings)
5. [Tips & Tricks](#tips--tricks)
6. [Troubleshooting](#troubleshooting)

---

## Getting Started

### Launching the Application

To start the web interface, run:

```bash
streamlit run src/f1_predict/web/app.py
```

The application will open in your default web browser at `http://localhost:8501`.

### First-Time Setup

1. The app uses the **Nebula Dark** theme by default
2. All settings are saved to `~/.f1_predict/settings.json`
3. Customize your preferences in the **Settings** page (see [Settings](#settings))

---

## Navigation

The main navigation menu is located in the left sidebar. Click any option to navigate to different sections:

- **Home** - Dashboard with quick stats and upcoming races
- **Predict** - ML-powered race predictions
- **Compare** - Compare drivers and teams
- **Analytics** - Comprehensive F1 analytics dashboard
- **Monitoring** - Model performance and drift detection
- **Explainability** - Understand predictions with SHAP analysis
- **Chat** - Ask questions about F1 in natural language
- **Settings** - Customize your preferences

---

## Features

### 🏠 Home Page

The home page provides a quick overview of key information:

- **Quick Stats**: Current season statistics at a glance
- **Upcoming Races**: Next 3 upcoming races with dates and locations
- **Quick Actions**: Navigation buttons to key features
- **AI Assistant**: Quick questions about F1

**Tips**:
- Click on an upcoming race to go to the Predict page
- Use the AI Assistant section for quick F1 queries

---

### 🏆 Predict Page

Make predictions for upcoming F1 races using our ML models.

**How to Use**:
1. Select a race from the dropdown
2. Choose your preferred prediction model:
   - **Ensemble** (recommended) - Combines multiple models
   - **XGBoost** - Fast and accurate
   - **LightGBM** - Gradient boosting
   - **Random Forest** - Robust and interpretable

3. Set your confidence threshold (0.0-1.0)
4. View predictions with:
   - Driver rankings
   - Win probabilities
   - Podium predictions
   - Feature importance visualization

**Confidence Threshold**:
- Higher threshold = more conservative predictions
- Shows only predictions above your confidence level
- Default: 0.7 (70%)

**Export Options**:
- Download predictions as CSV
- Export for external analysis
- Save predictions for reference

---

### 📊 Compare Page

Compare any drivers or teams across multiple metrics.

**Driver Comparison**:
1. Select two drivers from the dropdown
2. View side-by-side metrics:
   - Points scored
   - Wins and podiums
   - Win rate percentage
   - Qualifying performance
   - Circuit-specific statistics

3. Select circuits to see performance breakdown
4. View historical trends

**Team Comparison**:
1. Select two teams
2. Compare:
   - Constructor points
   - Total wins
   - Average points per race
   - Driver lineup statistics
   - Reliability metrics

**Filter Options**:
- Filter by season
- Filter by number of races
- View historical comparisons

---

### 📈 Analytics Dashboard

Deep-dive analysis of F1 data with interactive visualizations.

**Key Performance Indicators (KPIs)**:
- Races Analyzed
- Prediction Accuracy
- Average Confidence
- Data Quality Score

**Championship Standings**:
- **Drivers Championship**: Top drivers with points progression
- **Constructors Championship**: Team standings and competition

**Performance Visualizations**:
- **Win Rate**: Driver win percentages over time
- **Reliability**: Race completion rates and DNF trends
- **Qualifying vs Race**: Grid position vs. race result correlation

**Circuit Analysis**:
- Heatmap showing driver performance by circuit
- Identify which circuits suit which drivers
- Historical performance patterns

**Historical Trends**:
- Points progression for top teams
- Points distribution across drivers
- Seasonal trends and patterns

**Time Period Selection**:
- Last 5 Races
- Current Season
- Last 2 Seasons
- All Time

**Export Options**:
- Export standings as CSV
- Export performance metrics
- Export all analyzed data

---

### 🔍 Monitoring Page

Track model performance and detect data drift.

**Performance Metrics**:
- Real-time accuracy trends
- Prediction confidence distribution
- Model health scorecard
- Performance degradation alerts

**Drift Detection**:
- Feature drift status
- PSI (Population Stability Index) trends
- Distribution comparison
- Flagged features highlight

**Alert Management**:
- Create custom alert rules
- Set severity levels
- Configure delivery channels:
  - Console output
  - File logging
  - Email notifications
  - Slack integration

**Alert Configuration**:
1. Click on "Alert Configuration" tab
2. Enable desired alert channels
3. Provide recipient email or Slack webhook
4. Save configuration

---

### 🧠 Explainability Dashboard

Understand why the model makes specific predictions.

**SHAP Analysis**:
- Feature importance for each prediction
- Force plots showing feature contributions
- Waterfall plots for decision explanation
- Dependence plots for feature relationships

**Prediction Explanation**:
1. Select a race and driver
2. View base prediction value
3. See which features pushed prediction up/down
4. Understand model reasoning

**What-If Analysis**:
- Change feature values
- See how prediction would change
- Identify critical features
- Sensitivity analysis

**Tips**:
- Red features decrease prediction
- Blue features increase prediction
- Longer bars = stronger effect
- Use for model debugging and validation

---

### 💬 Chat Assistant

Ask natural language questions about F1 races and predictions.

**Suggested Queries**:
- "Who will win the next race?"
- "Compare Max Verstappen and Lewis Hamilton"
- "Show Red Bull's 2024 season performance"
- "What's the weather forecast for Monaco?"
- "Analyze the current championship standings"
- "Which driver has the best qualifying record?"

**How to Use**:
1. Type or select a suggested query
2. Wait for the AI response
3. View attached metrics and tables
4. Ask follow-up questions

**Chat Settings** (in sidebar):
- **AI Model**: Choose between GPT-4, Claude 3, or Local LLM
- **Creativity**: Adjust response creativity (0.0 = factual, 1.0 = creative)
- **Clear History**: Start a new conversation

**Tips**:
- Be specific for better results
- Ask about current season, historical data, or predictions
- View attached tables and metrics for data support

---

## Settings

Customize your application experience.

### General Settings
- **Color Theme**: Choose between Nebula Dark, Light, or High Contrast
- **Timezone**: Set your timezone for race times
- **Units**: Metric (km/h, °C) or Imperial (mph, °F)
- **Language**: Choose your preferred language
- **Auto-Refresh Interval**: How often to refresh data

### Prediction Settings
- **Default Model**: Model used by default (Ensemble, XGBoost, LightGBM, Random Forest)
- **Confidence Threshold**: Minimum confidence for displaying predictions (0.0-1.0)
- **Show Model Explanations**: Display feature importance with predictions
- **Show Feature Importance**: Which features contributed to prediction

### Display Settings
- **Chart Type**: Interactive (Plotly) or Static (Matplotlib)
- **Analytics Refresh Interval**: Auto-refresh frequency for analytics
- **Standings Page Size**: How many entries per page
- **Show Tooltips**: Display helpful hints
- **Analytics Auto-Refresh**: Automatically refresh dashboard

### API Configuration
- **API Endpoint**: Base URL for F1 data API
- **Rate Limit**: API requests per second (1-10)
- **Cache Duration**: How long to keep cached data (5-1440 minutes)
- **Request Timeout**: Maximum API wait time (5-300 seconds)
- **Enable API Caching**: Cache responses to reduce requests
- **Retry Attempts**: Failed request retries (0-5)

### Advanced Settings
- **Clear Cache**: Remove all cached data
- **Export Settings**: Download your settings as JSON
- **Import Settings**: Upload previously saved settings
- **Debug Logging**: Enable detailed logs
- **Performance Metrics**: Show internal performance stats
- **Developer Mode**: Advanced diagnostic features

---

## Tips & Tricks

### Performance Optimization
1. Disable auto-refresh if not needed
2. Use "Last 5 Races" filter for faster analytics
3. Enable caching in API settings
4. Clear cache regularly

### Better Predictions
1. Use Ensemble model for best accuracy
2. Check prediction confidence levels
3. View feature importance for context
4. Compare with historical data

### Data Exploration
1. Use Analytics dashboard for trends
2. Compare drivers/teams in different eras
3. Analyze circuit-specific performance
4. Check reliability metrics

### Troubleshooting Tips
1. Clear cache if data seems stale
2. Check API endpoint is correct
3. Verify internet connection for live data
4. Check rate limit if API is slow

---

## Troubleshooting

### Common Issues

**"No data available" message**
- Clear the cache (Settings → Advanced → Clear Cache)
- Check your internet connection
- Verify API endpoint is correct
- Try refreshing the page

**Slow predictions**
- Increase request timeout in Settings
- Clear cache to free up memory
- Use a faster model (XGBoost vs Ensemble)
- Reduce auto-refresh frequency

**Missing races**
- Ensure current season is selected
- Check race date filters
- Verify data was downloaded
- Try refreshing data

**Settings not saving**
- Check permissions on `~/.f1_predict/` directory
- Ensure you have write access to home directory
- Try exporting and re-importing settings
- Check disk space

### Getting Help

1. **Check Settings**: Ensure your configuration is correct
2. **View Logs**: Enable Debug Logging to see detailed information
3. **Reset to Defaults**: Reset Settings to factory defaults
4. **Check Documentation**: Review relevant feature documentation

---

## Keyboard Shortcuts

- **Ctrl/Cmd + R**: Refresh page
- **Sidebar Toggle**: Click menu icon to collapse/expand
- **Page Navigation**: Click sidebar options to navigate

---

## Data Sources

- **F1 Data**: Ergast API (ergast.com)
- **Real-Time Data**: Updated daily during season
- **Historical Data**: Complete F1 history from 2020+
- **Predictions**: ML models trained on historical data

---

## Privacy & Data

- Settings stored locally in `~/.f1_predict/settings.json`
- No personal data collected
- Data shared only with Ergast API for F1 information
- Cache stored locally on your machine

---

## System Requirements

- **Browser**: Modern browser (Chrome, Firefox, Safari, Edge)
- **Python**: 3.9+ (for running locally)
- **Memory**: 1GB minimum, 4GB recommended
- **Disk Space**: 500MB for data cache
- **Internet**: Required for live data

---

## Version Information

- **Application Version**: 1.0.0
- **Data Source**: Ergast API
- **Last Updated**: 2024

---

## Support & Feedback

For issues, feature requests, or feedback:
- Visit: https://github.com/jpequegn/f1-predict
- Report Issues: https://github.com/jpequegn/f1-predict/issues
- Discussions: https://github.com/jpequegn/f1-predict/discussions

---

## License

This project is licensed under the MIT License. See LICENSE file for details.

---

Happy predicting! 🏎️
