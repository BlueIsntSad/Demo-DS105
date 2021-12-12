import streamlit as st
import numpy as np
import pandas as pd
import pickle


############################ Custom function ############################
# Mã hoá các thuộc tính categories
def encode(X, ode, ohe):
    # Tạo bản sao để tránh ảnh hưởng dữ liệu gốc
    X_ = X.copy()
    
    # Biến categories có thứ tự
    X_[['Income group']] = ode.transform(X_[['Income group']])
    
    # Biến categories không thứ tự
    f_ohe = ohe.transform(X_[['Continent', 'WHO Region']])
    labels = np.array(ohe.categories_).ravel()
    cat_ohe = pd.DataFrame(f_ohe, columns=labels, index=X_.index)

    X_ = X_.drop(['Continent', 'WHO Region'], axis=1)
    X_ = pd.concat([X_, cat_ohe], axis=1)
    
    return X_

def prediction(samples, model):

    # Encode dữ liệu
    X_encode = encode(samples, ode, ohe)
    #data_encode = encode(data.iloc[:, :-1])

    # Predict
    return model.predict(X_encode)

############################ Layout pages ############################


def full_features_modeling(choice_input):
    st.subheader('Mô hình sử dụng toàn bộ thuộc tính')

    if choice_input == 'Dữ liệu mẫu':
        st.write('#### Sample dataset', data)

        # Chọn dữ liệu từ mẫu
        selected_indices = st.multiselect('Chọn mẫu từ bảng dữ liệu:', data.index)
        selected_rows = data.loc[selected_indices]

        st.write('#### Kết quả')

        if st.button('Dự đoán'):
            if not selected_rows.empty:
                X = selected_rows.iloc[:, :-1]
                pred = prediction(X, model_full)

                # Xuất ra màn hình
                results = pd.DataFrame({'Tỷ lệ tử vong dự đoán': pred,
                                        'Tỷ lệ tử vong thực tế': selected_rows.death_rate})
                st.write(results)
            else:
                st.error('Hãy chọn dữ liệu trước')

    elif choice_input == 'Tự chọn':
        st.write('Coming soon')


def select_features_modeling(choice_input):
    st.subheader('Mô hình sử dụng bộ thuộc tính chọn lọc')

    if choice_input == 'Dữ liệu mẫu':
        st.write('#### Sample dataset', data[selective_features + ['death_rate']])
        # Chọn dữ liệu từ mẫu
        selected_indices = st.multiselect('Chọn mẫu từ bảng dữ liệu:', data.index)
        selected_rows = data[selective_features + ['death_rate']].loc[selected_indices]

        st.subheader('Kết quả')
        if st.button('Dự đoán'):
            if not selected_rows.empty:
                X = selected_rows.iloc[:, :-1]
                pred = prediction(X, model_)

                # Xuất ra màn hình
                results = pd.DataFrame({'Tỷ lệ tử vong dự đoán (%)': pred,
                                        'Tỷ lệ tử vong thực tế (%)' : selected_rows.death_rate})
                st.write(results)
            else:
                st.error('Hãy chọn dữ liệu trước')

    elif choice_input == 'Tự chọn':
        X = data[selective_features]

        YChange = st.slider('Tỷ lệ tăng trưởng dân số hằng năm', X['Yearly Change'].min(), X['Yearly Change'].max(), X['Yearly Change'].mean())
        Fert = st.slider('Tỷ suất sinh', X['Fert. Rate'].min(), X['Fert. Rate'].max(), X['Fert. Rate'].mean())
        UPop = st.slider('Tỷ lệ dân số thành thị', X['Urban Pop %'].min(), X['Urban Pop %'].max(), X['Urban Pop %'].mean())
        HCI = st.slider('Chỉ số vốn con người - HCI', X['HCI'].min(), X['HCI'].max(), X['HCI'].mean())
        HE_GDP = st.slider("Mức chi tiêu cho y tế (% GDP)", X['Health_exp_pct_GDP'].min(), X['Health_exp_pct_GDP'].max(), X['Health_exp_pct_GDP'].mean())
        HE_public = st.slider('Tỷ lệ chi tiêu công cho y tế', X['Health_exp_public_pct'].min(), X['Health_exp_public_pct'].max(), X['Health_exp_public_pct'].mean())
        HE_external = st.slider('Mức chi tiêu cho y tế sử dụng nguồn tài trợ bên ngoài (% mức chi tiêu cho y tế)', X['External_health_exp_pct'].min(), X['External_health_exp_pct'].max(), X['External_health_exp_pct'].mean())
        Phy = st.slider('Số bác sĩ trên 1000 dân', X['Physicians_per_1000'].min(), X['Physicians_per_1000'].max(), X['Physicians_per_1000'].mean())
        Spec = st.slider('Số lượng đội ngũ y, bác sĩ chuyên khoa phẫu thuật trên 100 nghìn dân', X['Specialist_surgical_per_1000'].min(), X['Specialist_surgical_per_1000'].max(), X['Specialist_surgical_per_1000'].mean())
        Birth = st.slider('Tỷ lệ trẻ em dưới 5 tuổi được đăng ký giấy khai sinh', X['Completeness_of_birth_reg'].min(), X['Completeness_of_birth_reg'].max(), X['Completeness_of_birth_reg'].mean())

        IncomeG = st.selectbox('Phân loại nền kinh tế', X['Income group'].unique().tolist())
        Cont = st.selectbox('Châu lục', X['Continent'].unique().tolist())
        WHO = st.selectbox('Khu vực theo WHO', X['WHO Region'].unique().tolist())

        selected_data = {'Yearly Change': YChange,
                         'Fert. Rate': Fert,
                         'Urban Pop %': UPop,
                         'HCI': HCI,
                         'Income group': IncomeG,
                         'Health_exp_public_pct': HE_public,
                         'Health_exp_pct_GDP': HE_GDP,
                         'External_health_exp_pct': HE_external,
                         'Physicians_per_1000': Phy,
                         'Specialist_surgical_per_1000': Spec,
                         'Completeness_of_birth_reg': Birth,
                         'Continent': Cont,
                         'WHO Region': WHO}
        selected_rows = pd.DataFrame(selected_data, index=[0])

        st.subheader('Kết quả')
        if st.button('Dự đoán'):
            X = selected_rows
            pred = prediction(X, model_)

            # Xuất ra màn hình
            st.write('#### Tỷ lệ tử vong dự đoán là: ', round(pred[0],4) , '%')


############################ Main ############################

def main():
    st.title('Dự đoán tỷ lệ tử vong trên quy mô dân số do Covid-19')

    features_train = ['Toàn bộ thuộc tính',
                      'Bộ thuộc tính chọn lọc']
    choice_model = st.sidebar.selectbox('Mô hình huấn luyện trên:', features_train)

    input = ['Dữ liệu mẫu', 'Tự chọn']
    choice_input = st.sidebar.selectbox('Chọn kiểu nhập dữ liệu:', input)

    if choice_model == 'Toàn bộ thuộc tính':
        full_features_modeling(choice_input)

    elif choice_model == 'Bộ thuộc tính chọn lọc':
        select_features_modeling(choice_input)


############################ Run app ############################

if __name__ == '__main__':
    ## Load dataset
    data = pd.read_csv('data.csv',
                       index_col='Country (or dependency)')

    ## Tập các thuộc tính được chọn lọc
    selective_features = ['Yearly Change', 'Fert. Rate', 'Urban Pop %', 'HCI', 'Income group',
                          'Health_exp_public_pct', 'Health_exp_pct_GDP', 'External_health_exp_pct',
                          'Physicians_per_1000', 'Specialist_surgical_per_1000', 'Completeness_of_birth_reg',
                          'Continent', 'WHO Region']

    ## Load transformer
    with open("ode.pkl", 'rb') as file:
        ode = pickle.load(file)
    with open("ohe.pkl", 'rb') as file:
        ohe = pickle.load(file)

    ## Load model
    with open("model.pkl", 'rb') as file:
        model_ = pickle.load(file)
    with open("model_full.pkl", 'rb') as file:
        model_full = pickle.load(file)


    main()