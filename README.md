<div align="center">

```
██╗  ██╗ ██████╗ ██╗   ██╗███████╗███████╗    ██████╗ ██████╗ ██╗ ██████╗███████╗
██║  ██║██╔═══██╗██║   ██║██╔════╝██╔════╝    ██╔══██╗██╔══██╗██║██╔════╝██╔════╝
███████║██║   ██║██║   ██║███████╗█████╗      ██████╔╝██████╔╝██║██║     █████╗  
██╔══██║██║   ██║██║   ██║╚════██║██╔══╝      ██╔═══╝ ██╔══██╗██║██║     ██╔══╝  
██║  ██║╚██████╔╝╚██████╔╝███████║███████╗    ██║     ██║  ██║██║╚██████╗███████╗
╚═╝  ╚═╝ ╚═════╝  ╚═════╝ ╚══════╝╚══════╝    ╚═╝     ╚═╝  ╚═╝╚═╝ ╚═════╝╚══════╝

██████╗ ██████╗ ███████╗██████╗ ██╗ ██████╗████████╗██╗ ██████╗ ███╗   ██╗
██╔══██╗██╔══██╗██╔════╝██╔══██╗██║██╔════╝╚══██╔══╝██║██╔═══██╗████╗  ██║
██████╔╝██████╔╝█████╗  ██║  ██║██║██║        ██║   ██║██║   ██║██╔██╗ ██║
██╔═══╝ ██╔══██╗██╔══╝  ██║  ██║██║██║        ██║   ██║██║   ██║██║╚██╗██║
██║     ██║  ██║███████╗██████╔╝██║╚██████╗   ██║   ██║╚██████╔╝██║ ╚████║
╚═╝     ╚═╝  ╚═╝╚══════╝╚═════╝ ╚═╝ ╚═════╝   ╚═╝   ╚═╝ ╚═════╝ ╚═╝  ╚═══╝
```

### 🏠 *because someone had to teach a computer what a house is worth*

<br>

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Tkinter](https://img.shields.io/badge/Tkinter-GUI-4B8BBE?style=for-the-badge&logo=python&logoColor=white)
![Status](https://img.shields.io/badge/Status-It_Actually_Works-27AE60?style=for-the-badge)
![R²](https://img.shields.io/badge/R²_Score-0.96-FF6B6B?style=for-the-badge)

<br>

> *"My cousin paid ₹85 lakhs for a 2BHK. I made an ML model. We are not the same."*

</div>

---

<br>

## okay so here's the story

it started as a college assignment. the kind where the prof says *"build something with ML"* and gives you exactly zero guidance on what that means.

most of my classmates downloaded a kaggle notebook, changed the title, and submitted it.

i didn't want to do that.

i wanted to build something that actually **runs**. something i could open my laptop, demo in an interview, and not feel like a fraud doing it. something where if someone asked *"how does the preprocessing work"* i could actually answer without sweating through my shirt.

so i spent a weekend on it. three cups of chai, one all-nighter, and a concerning amount of Stack Overflow. and this is what came out.

it predicts house prices. it has a real GUI. the model actually learned something (R² of 0.96, not bad for a first build). and i'm weirdly proud of it.

<br>

---

<br>

## 👀 what this thing actually does

you open the app. you fill in some details about a house — area, bedrooms, location, how old it is, whether it's furnished. you click one button. it tells you what that house is probably worth in rupees.

that's it. no cloud. no API calls. no subscription. just python doing math on your laptop.

under the hood it's a full ML pipeline — data loading, cleaning, encoding categoricals, scaling, training, evaluating, saving the model to disk, and loading it back. the whole thing. not a jupyter notebook you have to run cell by cell. actual `.py` files that you run like a normal human being.

<br>

---

<br>

## 🧠 the brains behind it

```
your CSV  →  clean it  →  encode it  →  scale it  →  train it  →  save it  →  load it  →  predict it
```

yeah i know that looks simple. that's the point. the complexity is in the details:

- **missing values?** handled. numeric columns get median-imputed (median, not mean — because outliers are real and they will ruin your life), categorical columns get mode-imputed.
- **"Urban" and "Furnished" aren't numbers?** Label Encoding turns them into numbers. the same encoder that trained is the same one that runs in the GUI. same transformation, every time.
- **features at wildly different scales?** StandardScaler. because feeding raw square footage alongside a "0 or 1" bedroom count into a regression is asking for trouble.
- **model saved with pickle** so you don't retrain every time you open the app. train once, predict forever.

the architecture is OOP — `DataLoader`, `DataPreprocessor`, `ModelTrainer`, `Visualizer`, `ModelSerializer` — each class does one thing. i wrote it this way so i could actually explain it if someone asks. because they will ask.

<br>

---

<br>

## 📊 model performance (the numbers that matter in interviews)

| what we measured | what we got | what it means in plain english |
|---|---|---|
| **R² Score** | **0.9599** | the model explains ~96% of price variation. yes really. |
| **MAE** | ₹ 7,42,344 | on average, off by about 7.4 lakhs |
| **RMSE** | ₹ 9,32,132 | punishes big misses more. still under a lakh per predicted crore |

> quick test i ran: Urban, 1800 sq ft, 3BHK, Semi-Furnished, 5 years old → **₹ 87,37,458**
>
> checked with some real listings in similar areas. it's... not wrong. 

<br>

---

<br>

## 📁 project structure (and why it's set up this way)

```
House_Price_Prediction/
│
├── 📂 dataset/
│   └── housing.csv           ← 100 rows. area, rooms, location, price. clean and ready.
│
├── 📂 models/
│   └── trained_model.pkl     ← the brain. model + encoders + scaler, all in one file.
│
├── 📂 charts/                ← auto-generated after training. 4 plots, no manual work.
│   ├── correlation_heatmap.png
│   ├── price_distribution.png
│   ├── area_vs_price.png
│   └── actual_vs_predicted.png
│
├── 🐍 train_model.py         ← run this first. does everything: load → clean → train → save.
├── 🖥️  app.py                ← the GUI. open this to actually predict house prices.
├── 📋 requirements.txt       ← four libraries. that's it.
└── 📖 README.md              ← you're reading it
```

two files to care about. `train_model.py` does the ML work. `app.py` is what you actually show people.

<br>

---

<br>

## ⚡ getting this running (for real, step by step)

**step 0 — clone it**
```bash
git clone https://github.com/your-username/house-price-prediction.git
cd house-price-prediction
```

**step 1 — virtual environment (don't skip this, i'm serious)**
```bash
python -m venv venv

# if you're on windows
venv\Scripts\activate

# if you're on mac/linux
source venv/bin/activate
```

**step 2 — install the four libraries you need**
```bash
pip install -r requirements.txt
```

that installs: `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`. tkinter comes with python already. you're done.

**step 3 — train the model**
```bash
python train_model.py
```

you'll see it print dataset stats, log the preprocessing steps, show the evaluation metrics, tell you where it saved the charts and the model. takes about 3 seconds.

**step 4 — open the app**
```bash
python app.py
```

window opens. fill in the fields. click predict. done.

> **if you skip step 3** and run `app.py` directly, it auto-trains before opening the window. i added that because i knew i'd forget to train it before demoing to someone.

<br>

> **linux users** — if tkinter throws an error:
> ```bash
> sudo apt-get install python3-tk
> ```

<br>

---

<br>

## 🎨 the charts it generates

after training, four charts get saved to `charts/` automatically. here's what they show and why they're actually useful:

**correlation heatmap** — which features actually move the price needle. spoiler: area and location matter a lot. house age matters less than you'd think.

**price distribution** — a histogram of all 100 house prices with a KDE curve on top. useful for spotting if your dataset is weirdly skewed.

**area vs price (scatter)** — coloured by location. the Urban/Suburban/Rural gap is visible immediately. this is the kind of chart you put in your slide deck.

**actual vs predicted** — the one interviewers like. dots close to the diagonal line = good model. mine are close. 

<br>

---

<br>

## 🗃️ dataset columns (quick reference)

| column | type | range / values |
|---|---|---|
| `Area` | int | sq ft, 700 – 3200 |
| `Bedrooms` | int | 1 – 5 |
| `Bathrooms` | int | 1 – 5 |
| `Floors` | int | 1 – 3 |
| `Parking` | int | 0 – 4 spaces |
| `Location` | str | Urban / Suburban / Rural |
| `Furnishing` | str | Furnished / Semi-Furnished / Unfurnished |
| `HouseAge` | int | 1 – 25 years |
| `Price` | int | ₹ target variable |

100 rows. synthetic but realistic numbers. built to reflect actual Indian housing market patterns — Urban > Suburban > Rural, furnished adds ~30% premium, newer = pricier.

<br>

---

<br>

## 💬 things i'd do differently (or will add later)

look, Linear Regression is a starting point. it's not the best algorithm for this. it's the most *explainable* algorithm for this, which matters more in college projects and interviews. you can literally say *"the coefficient on Area means each additional square foot adds X rupees to the predicted price"* and watch the interviewer nod.

but if i keep building this:

- [ ] **Random Forest / XGBoost** — compare with LR side by side. RF will win.
- [ ] **K-Fold cross-validation** — stop relying on one random 80/20 split
- [ ] **SHAP values** — show *why* the model predicted what it predicted, per house
- [ ] **Streamlit web app** — replace Tkinter so it runs in a browser and i can share a link
- [ ] **real data scraping** — 99acres or MagicBricks with BeautifulSoup
- [ ] **hyperparameter tuning** — GridSearchCV on the whole pipeline
- [ ] **more cities** — right now it's Urban/Suburban/Rural. should be actual city names.

<br>

---

<br>

## 🤝 if you want to use this

go for it. fork it, clone it, submit it for your college project (just understand what it does first — your professor will ask). if you improve it, open a PR. if you find a bug, open an issue.

if this helped you get an internship, that would genuinely make my day. let me know.

<br>

---

<br>

<div align="center">

built with Python, scikit-learn, and mild sleep deprivation

*if the model is wrong about your house price, please don't @ me*

---

⭐ **star this if it helped you** — it costs nothing and means something

</div>
