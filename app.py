import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import joblib


st.set_page_config(page_title="Analisis Produk Sephora")

file_path_clf = 'gnb.pkl'
with open(file_path_clf, 'rb') as f:
    clf = joblib.load(f)

df_data_before_map = 'data_before_map.csv' #dataset cleaned sebelum di mapping
df_data_after_map = 'Data_Cleaned.csv' #dataset cleaned setelah di mapping
df_data_gnb = 'sephora_restored.csv' #dataset cleaned yang mengembalikan kolom product_name dan kolom primary_category
df_data_ori = 'product_info.csv' #dataset awal/asli


product_description = {
    "Fragrance": "Produk yang digunakan untuk memberikan aroma yang menyenangkan pada tubuh, mencakup produk parfum, cologne, body mist, eau de toilette, eau de parfum, dan eau de cologne.",
    "Bodycare":"Produk yang digunakan untuk membersihkan, melembabkan, dan melindungi kulit tubuh, seperti body wash, body lotion, body scrub, body butter, soap, bath oil, bath salt, deodorant, dan sunscreen.",
    "Haircare": "Ptoduk yang digunakan untuk membersihkan, merawat, dan menata rambut, yaitu shampoo, conditioner, masker rambut, leave-in conditioner, hair oil, hairspray, dry shampoo, hair gel, dan mousse.",
    "Skincare": "Produk yang digunakan untuk membersihkan, merawat, dan melindungi kulit wajah, mencakup cleanser, toner, moisturizer, masker, serum, sunscreen, eye cream, lip balm, exfoliator, dan toner.",
    "Makeup": "Produk yang digunakan untuk mempercantik wajah dan kuku, seperti foundation, concealer, powder, blush, bronzer, highlighter, eyeshadow, eyeliner, mascara, lipstick, lip gloss, lip liner, dan nail polish.",
    "Tools": "Produk yang digunakan untuk membantu dalam proses makeup dan penataan rambut, mencakup makeup brushes, beauty sponges, eyelash curlers, tweezers, nail clippers, nail files, hair brushes, hair ties, dan hair clips.",
    "Other": "Kategori produk tambahan seperti  gift sets, travel sets, men's grooming products, dan home fragrance."
}

st.title('Analisis Sentimen Ulasan Pengguna dalam Menentukan Produk Rekomendasi di Sephora')

selected_page = st.sidebar.selectbox(
    "Select Page",
    ["Dashboard", "Visualisasi Analysis", "Kesimpulan Rekomendasi", "Data Prediction"]
)

if selected_page == "Dashboard":
    data_gnb = pd.read_csv(df_data_gnb)
    data_ori = pd.read_csv(df_data_ori)
    data_before_map = pd.read_csv(df_data_before_map)
    data_after_map = pd.read_csv(df_data_after_map)

    st.image('sephora.png', caption='sephora', use_column_width=True)

    st.subheader("Tujuan")
    st.write("Menentukan produk sephora yang populer atau direkomendasikan melalui analisis sentimen ulasan pelanggan/pengguna Sephora sehingga dapat meningkatkan pengalaman dan kepuasan pelanggan/pengguna Sephora.")

    st.subheader("Algoritma Modelling")
    st.write("Analisis ini menggunakan pendekatan prediktif digunakan dalam analisis sentimen ulasan pengguna untuk menentukan produk rekomendasi di Sephora.")

    st.subheader("Dataset Original")
    st.write(data_ori)
    st.write('Dataset diatas merupakan dataset original dair sephora, yang mana dataset tersebut diambil dari kaggle. Dataset ini berisi data produk yang dijual di sephora baik dari brand sephora collection bahkan dari brand lain.')

    st.subheader('Dataset Preprocessed Before Mapping')
    st.write(data_before_map)
    st.write('Dataset ini merupakan dataset yang telah di preprocessing dan di cleaning, yang melalui tahap data preparation, namun hanya sampai tahap data Cleaning, sehingga dataset ini belum memasuki tahap mapping hingga evaluation. Dataset ini berisi data produk yang dijual di sephora baik dari brand sephora collection bahkan dari brand lain.')

    st.subheader('Dataset Preprocessed After Mapping')
    st.write(data_gnb)
    st.write('Dataset ini merupakan dataset yang telah di preprocessing dan di cleaning, yang sudah melalui tahap data preparation hingga data evaluation. Dataset ini berisi data produk yang dijual di sephora baik dari brand sephora collection bahkan dari brand lain.')

    st.subheader("Kategori Produk yang Ditawarkan")
    selected_product = st.selectbox('Select Product', list(product_description.keys()))
    st.markdown(f"{selected_product}: {product_description[selected_product]}")

elif selected_page == "Visualisasi Analysis":
    data_gnb = pd.read_csv(df_data_gnb)
    data_before_map = pd.read_csv(df_data_before_map)
    data_after_map = pd.read_csv(df_data_after_map)
    
    # =============================== Kategori Produk ===============================

    st.subheader("Data Distribusi Kategori Produk")
    main_categories = ['Fragrance', 'Bath & Body', 'Hair', 'Skincare', 'Makeup', 'Tools & Brushes']
    data_gnb['primary_category'] = data_gnb['primary_category'].apply(lambda x: x if x in main_categories else 'Other')

    fig_kategori, ax = plt.subplots(figsize=(10, 6))
    data_gnb['primary_category'].value_counts().plot(kind='bar', ax=ax)
    ax.set_title('Distribusi Kategori Produk pada Sephora')
    ax.set_xlabel('Kategori Produk')
    ax.set_ylabel('Jumlah Produk')
    st.pyplot(fig_kategori)

    st.markdown('**Hasil**')
    st.write('Diagram diatas menunjukkan bagaimana distribusi kategori produk yang ditawarkan pada sephora. Dapat dilihat bahwa di sephora menyediakan beberapa kategori produk, mulai dari skincare, makeup, fragrances, dan lain sebagainya. Kategori produk yang paling banyak tersedia di sephora adalah skincare, dapat dilihat dari diagram batang pada skincare yang lebih tinggi dibanding kategori lainnya. sehingga dari hasil tersebut dapat ditarik kesimpulan bahwa produk yang lebih banyak peminatnya adalah produk yang termasuk kategori skincare.')

    # =============================== Perbandingan Sentimen Positif/Negatif Sesuai Rating ===============================

    positive_data = data_gnb[data_gnb['rating'] >= 3]
    negative_data = data_gnb[data_gnb['rating'] < 3]

    positive_count = len(positive_data)
    negative_count = len(negative_data)

    st.subheader('Proporsi Sentimen Produk Berdasarkan Rating')
    fig, ax = plt.subplots()
    ax.set_title('Perbandingan Sentimen Positif/Negatif Sesuai Rating')
    ax.pie([positive_count, negative_count], labels=['Positif', 'Negatif'], autopct='%1.1f%%', colors=['red', 'lightgrey'])
    st.pyplot(fig)

    st.markdown('**Hasil**')
    st.write('Diagram diatas menunjukkan bagaimana rating mempengaruhi sentiment pengguna, apakah produk tersebut memiliki sentiment positif atau negatif. jika rata-rata raing dari sebuah produk lebih kecil dari 3, maka negatif. sebaliknya, jika rata-rata rating dari sebuah produk adalah lebih dari atau sama dengan 3, maka positif. dari hasil diagram dapat dilihat bahwa produk sephora lebih banyak memiliki sentiment positif, sehingga dapat disimpulkan bahwa lebih banyak pengguna yang menilai positif akan produk tersebut dengan memberikan rating terhadap produk yang digunakan.')


    # =============================== Kepopuleran sesuai Loves Count ===============================

    st.subheader('Distribusi Kepopuleran Produk pada Sephora sesuai Loves Count')
    popularity_labels = ['Tidak Populer', 'Cukup Populer', 'Populer', 'Sangat Populer', 'Viral']
    popularity_counts = [((data_before_map['loves_count'] < 100).sum()),
                        ((data_before_map['loves_count'] >= 100) & (data_before_map['loves_count'] < 1000)).sum(),
                        ((data_before_map['loves_count'] >= 1000) & (data_before_map['loves_count'] < 10000)).sum(),
                        ((data_before_map['loves_count'] >= 10000) & (data_before_map['loves_count'] < 50000)).sum(),
                        ((data_before_map['loves_count'] >= 50000)).sum()]

    fig_populer, ax = plt.subplots(figsize=(8, 6))
    ax.bar(popularity_labels, popularity_counts, color='skyblue')
    ax.set_title('Produk Paling Populer di Sephora')
    ax.set_xlabel('Kategori Kepopuleran')
    ax.set_ylabel('Jumlah Produk')
    st.pyplot(fig_populer)
    
    st.markdown('**Hasil**')
    st.write('Diagram diatas menunjukkan distribusi kepopuleran produk pada sephora, yang menampilkan produk yang paling populer, dimana kepopuleran ini dinilai dari loves_count dari pengguna. semakin banyak loves_count yang dimiliki sebuah produk artinya semakin populer produk tersebut. Dari diagram dapat dilihat bahwa produk yang disediakan di sephora dinilai populer, dapat dilihat dari diagram batang pada kategori penilaian yang lebih tinggi di bagian populer. sehingga dapat disimpulkan bahwa produk sephora adalah populer bagi penggunanya.')

    # =============================== Distribusi Reviews Pengguna ===============================

    def kategori_penilaian(reviews):
        if reviews < 50:
            return 'Penilaian Buruk'
        elif reviews < 500:
            return 'Penilaian Sedang'
        else:
            return 'Penilaian Baik'

    st.subheader('Proporsi Kategori Penilaian Produk sesuai Reviews')
    data_before_map['Kategori Penilaian'] = data_before_map['reviews'].apply(kategori_penilaian)
    count_per_kategori = data_before_map['Kategori Penilaian'].value_counts()

    fig_reviews, ax = plt.subplots()
    ax.set_title('Penilaian Produk')
    ax.pie(count_per_kategori, labels=count_per_kategori.index, autopct='%1.1f%%', colors=['red', 'grey', 'lightgrey'])
    st.pyplot(fig_reviews)
    
    st.markdown('**Hasil**')
    st.write('Diagram diatas menunjukkan bagaimana distribusi penilaian dari pengguna di sephora, yang dinilai sesuai reviews sebuah produk. semakin banyak yang mereview sebuah produk, artinya semakin baik penilaian terhadap produk tersebut. Dari hasil diagram dapat dilihat bahwa penilaian menurut reviews lebih banyak menunjukkan bahwa produk sephora memiliki penilaian sedang. artinya reviews tidak sedikit dan tidak banyak.')

    # =============================== Top 10 Brand ===============================

    st.subheader('10 Brand dengan Jumlah Produk Terbanyak')
    brand_distribution = data_gnb['brand_name'].value_counts()
    top_brands = brand_distribution.head(10)

    fig_brand, ax = plt.subplots(figsize=(10, 6))
    top_brands.plot(kind='barh', color='skyblue', ax=ax)
    ax.set_title('Distribusi Top 10 Brand')
    ax.set_xlabel('Jumlah Produk')
    ax.set_ylabel('Brand Name')
    plt.tight_layout()
    st.pyplot(fig_brand)

    st.markdown('**Hasil**')
    st.write('Diagram diatas menunjukkan 10 brand paling populer di sephora. Dari hasil diagram dapat dilihat bahwa brand yang memiliki produk terbanyak adalah brand SEPHORA COLLECTION yaitu sebanyak lebih dari 350 produk sesuai diagram yang ditampilkan diatas.')

    #============================== Brand Sephora Collection ===============================

    st.markdown('**Produk SEPHORA COLLECTION**')
    sephora_data = data_gnb[data_gnb['brand_name'] == 'SEPHORA COLLECTION']
    st.write(sephora_data[['brand_name', 'product_name', 'primary_category']])
    st.write('Berikut ini adalah produk yang disediakan oleh brand SEPHORA COLLECTION di sephora')

    # =============================== 10 Produk dengan Sentimen Positif Sesuai Rating Terbanyak ===============================

    st.subheader('Produk dengan Sentimen Positif Sesuai Rating Terbanyak')
    positive_data = data_before_map[data_before_map['rating'] >= 3]
    top_10_positive_products = positive_data.sort_values(by='rating', ascending=False).head(10)

    fig, ax = plt.subplots()
    ax.barh(top_10_positive_products['product_name'], top_10_positive_products['rating'], color='skyblue')
    ax.set_title('Produk dengan Sentimen Positif Sesuai Rating Terbanyak')
    ax.set_xlabel('Rating')
    ax.set_ylabel('Nama Produk')
    ax.invert_yaxis() 
    st.pyplot(fig)

    st.markdown('**Hasil**')
    st.write('Diagram diatas menunjukkan 10 produk di sephora yang memiliki sentiment positif. sentiment positif ini dinilai dari banyaknya rating sebuah produk, dimana akan termasuk positif, jika rata-rata rating lebih atau sama dengan 3. dari hasil diagram dapat dilihat bahwa dari 10 contoh produk yang diambil, semua produk tersebut memiliki rating positif dengan rata-rata rating adalah 5.0, artinya produk di sephora dapat dikatakan memiliki sentiment positif dari pengguna yang mensubmit rating untuk produk yang digunakannya.')


    # =============================== Produk yang  Paling Populer sesuai Loves Count ===============================

    st.subheader('Top 10 Produk Paling Populer Berdasarkan Loves Count')
    most_popular_products = data_before_map.groupby(['product_name', 'brand_name'])['loves_count'].sum().reset_index(name='total_loves').sort_values(by='total_loves', ascending=False)
    top_10_popular_products = most_popular_products.head(10)

    fig, ax = plt.subplots()
    ax.barh(top_10_popular_products['product_name'] + " (" + top_10_popular_products['brand_name'] + ")", top_10_popular_products['total_loves'], color='skyblue')
    ax.set_title('10 Produk Paling Populer')
    ax.set_xlabel('Total Loves Count')
    ax.set_ylabel('Nama Produk (Brand)')
    st.pyplot(fig)
    
    st.markdown('**Hasil**')
    st.write('Diagram diatas menunjukkan 10 produk paling populer di sephora. Produk akan dinilai semakin populer apabila semakin banyak loves_count yang diberikan pengguna. Dari hasil diagram dapat dilihat bahwa produk yang paling populer adalah "Midnight Recovery Concentrate Moisturizing Face Oil" dari brand "Kiehl,s Since 1851" yaitu yang memiliki loves_count sebanyak lebih dari 80000.')

    # =============================== Produk dengan Penilaian Baik sesuai Reviews ===============================

    most_reviewed_products = data_before_map.groupby('product_name')['reviews'].sum().reset_index(name='total_reviews').sort_values(by='total_reviews', ascending=False)
    top_10_reviewed_products = most_reviewed_products.head(10)

    st.subheader('Top 10 Produk dengan Ulasan Paling Banyak')
    fig, ax = plt.subplots()
    ax.barh(top_10_reviewed_products['product_name'], top_10_reviewed_products['total_reviews'], color='skyblue')
    ax.set_xlabel('Jumlah Ulasan')
    ax.set_ylabel('Nama Produk')
    ax.set_title('Top 10 Produk dengan Ulasan Paling Banyak')
    st.pyplot(fig)

    st.markdown('**Hasil**')
    st.write('Diagram diatas menunjukkan 10 produk yang memiliki reviews paling banyak. dimana semakin banyak reviews dari pengguna artinya produk tersebut semakim dinilai baik atau mendapat penilaian baik. Dari diagram diatas dapat dilihat bahwa produk yang memiliki ulasan atau review terbanyak adalah "Midnight Recovery Concentrate Moisturizing Face Oil" yaitu yang telah di review sebanyak lebih dari 2000.')

elif selected_page == "Kesimpulan Rekomendasi":
    data_gnb = pd.read_csv(df_data_gnb)
    data_before_map = pd.read_csv(df_data_before_map)
    data_after_map = pd.read_csv(df_data_after_map)

    feature_options = ['Terpopuler (loves_count)', 'Penilaian Terbaik (reviews)', 'Sentiment Positif (rating)' ]
    selected_recommend = st.selectbox('Produk Rekomendasi Menurut:', feature_options)

    # =============================== Produk Rekomendasi dilihat dari tingkat populer sesuai Loves Count ===============================

    if selected_recommend == 'Terpopuler (loves_count)':
        popular_products = data_before_map.groupby(['product_name', 'brand_name', 'primary_category'])['loves_count'].sum().reset_index()
        most_popular_product = popular_products.sort_values(by='loves_count', ascending=False).iloc[0]

        st.subheader('Produk Paling Populer')
        most_popular_product_df = pd.DataFrame({
            "Nama Produk": [most_popular_product['product_name']],
            "Nama Merek": [most_popular_product['brand_name']],
            "Kategori Produk": [most_popular_product['primary_category']],
            "Total Loves Count": [most_popular_product['loves_count']]
        })
        st.table(most_popular_product_df)

        st.markdown("**Kesimpulan**")
        st.write(f"Produk paling populer, yaitu produk rekomendasi dilihat dari tingkat populer sesuai Loves Count adalah {most_popular_product['product_name']} \
                dari merek {most_popular_product['brand_name']} dan termasuk kategori {most_popular_product['primary_category']}, \
                dengan total loves count sebanyak {most_popular_product['loves_count']}.")


    # =============================== Produk Rekomendasi dilihat dari tingkat penilaian terbaik sesuai Reviews ===============================

    elif selected_recommend == 'Penilaian Terbaik (reviews)':
        total_reviews_per_product = data_before_map.groupby('product_name')['reviews'].sum().reset_index()
        most_reviewed_product = total_reviews_per_product.sort_values(by='reviews', ascending=False).iloc[0]
        product_info = data_before_map[data_before_map['product_name'] == most_reviewed_product['product_name']].iloc[0]

        st.subheader('Produk dengan Penilaian Terbaik')
        most_reviewed_product_df = pd.DataFrame({
            "Nama Produk": [most_reviewed_product['product_name']],
            "Nama Merek": [product_info['brand_name']],
            "Kategori Produk": [product_info['primary_category']],
            "Total Ulasan": [most_reviewed_product['reviews']]
        })
        st.table(most_reviewed_product_df)

        st.markdown("**Kesimpulan**")
        st.write(f"Produk rekomendasi dengan penilaian terbaik sesuai review pengguna adalah {most_reviewed_product['product_name']} dari merek {product_info['brand_name']} \
                yang termasuk  kategori {product_info['primary_category']}, dengan total ulasan sebanyak {most_reviewed_product['reviews']}.")


    # =============================== Produk Rekomendasi dilihat dari tingkat sentimen positif sesuai Rating ===============================

    elif selected_recommend == 'Sentiment Positif (rating)':
        positive_data = data_before_map[data_before_map['rating'] >= 3]

        most_positive_product = positive_data.groupby(['product_name', 'brand_name', 'primary_category'])['rating'].mean().reset_index()
        most_positive_product = most_positive_product.sort_values(by='rating', ascending=False).iloc[0]

        st.subheader('Produk dengan Sentimen Positif Tertinggi')
        most_positive_product_df = pd.DataFrame({
            "Nama Produk": [most_positive_product['product_name']],
            "Nama Merek": [most_positive_product['brand_name']],
            "Kategori Produk": [most_positive_product['primary_category']],
            "Rating Rata-rata": [most_positive_product['rating']]
        })
        st.table(most_positive_product_df)

        st.markdown("**Kesimpulan**")
        st.write(f"Produk rekomendasi dengan sentimen positif tertinggi sesuai rating pengguna adalah {most_positive_product['product_name']} dari merek {most_positive_product['brand_name']} \
                yang termasuk dalam kategori {most_positive_product['primary_category']}, dengan rating rata-rata sebesar {most_positive_product['rating']}.")



elif selected_page == "Data Prediction":
    data_before_map = pd.read_csv(df_data_before_map)
    st.subheader("Prediksi Produk Rekomendasi")

    product_options = data_before_map['product_name'].unique()
    selected_product_name = st.selectbox('product_name', product_options)

    selected_brand_name = data_before_map[data_before_map['product_name'] == selected_product_name]['brand_name'].iloc[0]

    min_rating_released = int(data_before_map['rating'].min())
    max_rating_released = int(data_before_map['rating'].max())
    selected_rating_released = st.slider('rating', min_rating_released, max_rating_released)

    min_loves_count = int(data_before_map['loves_count'].min())
    max_loves_count = int(data_before_map['loves_count'].max())
    selected_loves_count = st.slider('loves_count', min_loves_count, max_loves_count)

    min_reviews = int(data_before_map['reviews'].min())
    max_reviews = int(data_before_map['reviews'].max())
    selected_reviews = st.slider('reviews', min_reviews, max_reviews)

    if selected_rating_released >= 3 and selected_loves_count >= 1000 and selected_reviews >= 50:
        product_recommended = "Direkomendasikan"
    else:
        product_recommended = "Tidak Direkomendasikan"

    result = pd.DataFrame({
        'product_name': [selected_product_name],
        'brand_name': [selected_brand_name],
        'rating': [selected_rating_released],
        'loves_count': [selected_loves_count],
        'reviews': [selected_reviews],
        'product_recommended': [product_recommended]
    })

    st.subheader("Tabel Hasil Prediksi:")
    st.write(result)  # Menampilkan tabel hasil prediksi

    # Menampilkan kesimpulan
    st.subheader("Kesimpulan:")
    st.write(f"Produk: {selected_product_name} dari brand: {selected_brand_name} dengan rating: {selected_rating_released}, loves_count: {selected_loves_count}, dan reviews: {selected_reviews} merupakan produk yang: {product_recommended}")

