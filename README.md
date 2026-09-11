# 🛍️ Promoting Local and Domestic brands by Using AI-Powered E-commerce with Product Recommendation

A deep learning-powered fashion recommendation web application built with **PyTorch**, **ResNet50**, and **Streamlit**. The system extracts visual feature embeddings from catalog images to recommend similar fashion products using Nearest Neighbors similarity search.

---

## 📌 Features

* **Feature Extraction:** Powered by a pre-trained **ResNet50** convolutional neural network (`torchvision`).
* **Similarity Search:** Uses **scikit-learn's `NearestNeighbors`** (Euclidean distance) to find visual matches.
* **Interactive UI:** Simple, real-time image upload and recommendation display using **Streamlit**.
* **Cloud-Ready:** Lightweight CPU-optimized deployment configuration for Streamlit Cloud.

---

## 🛠️ Tech Stack

* **Language:** Python 3.14.7
* **Framework:** PyTorch (`torch`, `torchvision`)
* **Web UI:** Streamlit
* **ML / Math Tools:** scikit-learn, NumPy, Pillow, tqdm

---

## Deployment link : https://finalyrprojectdypuambi-snds6bnukk7dqmjvhlipet.streamlit.app/
## 📁 Project Structure

```text
├── .github/
│   └── workflows/        # GitHub Actions CI/CD workflows
├── images/               # Dataset directory containing catalog images
├── uploads/              # Temporary storage for user-uploaded queries
├── app.py                # Pipeline script to generate image embeddings
├── main.py               # Streamlit application entry point
├── embeddings.pkl        # Precomputed feature vectors
├── filenames.pkl         # Dataset file paths mapping
├── requirements.txt      # Project dependencies
└── README.md             # Project documentation
