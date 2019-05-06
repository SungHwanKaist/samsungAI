# coding: utf-8

# In[1]:


import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
import tensorflow as tf
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split  # to create validation data set
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder

from collections import namedtuple


# In[2]:
def region_roadtype():
    raw_data = pd.read_csv('./dataset/Kor_Train_traffic_accident_fatalities.csv',
                           encoding="cp949")

    pre_data = raw_data.drop(columns = ['발생지시군구', '도로형태_대분류'])

    pre_df = raw_data.drop(columns = list(pre_data))
    pre_df = pre_df.reset_index(drop=True)

    for i in range(pre_df.shape[0]):
        if pre_df.loc[i][1] == '단일로':
            pass
        elif pre_df.loc[i][1] == '교차로':
            pass
        else:
            pre_df.at[i, '도로형태_대분류'] = 0

    df = pre_df[pre_df['도로형태_대분류'] != 0]

    region_list = df['발생지시군구'].tolist()
    region_list = list(set(region_list))
    
    region_dict = dict()

    s = dict()
    for region in region_list:
        df_reg = df[df['발생지시군구'] == region]
        s[region] = df_reg['도로형태_대분류'].value_counts(1)['단일로']
        
    region_dict['발생지시군구'] = s
    return region_dict


def transfrom_dict(opt):
    transfrom_dict_car_car = {
        '주야': {'야간': 0, '주간': 1},

        '요일': {'월': 0, '화': 0, '수': 0, '목': 0, '금': 1, '토': 1, '일': 1},

        '사고유형_중분류': {'기타': 2,
                     '정면충돌': 9,
                     '추돌': 14,
                     '측면직각충돌': 16,
                     '측면충돌': 16,
                     '후진중충돌': 2},

        '법규위반': {'과로': 3,
                 '과속': 1,
                 '교차로 통행방법 위반': 2,
                 '기타(운전자법규위반)': 3,
                 '보행자 보호의무 위반': 3,
                 '부당한 회전': 3,
                 '서행 및 일시정지위반': 3,
                 '신호위반': 8,
                 '안전거리 미확보': 9,
                 '안전운전 의무 불이행': 10,
                 '앞지르기 금지위반': 3,
                 '앞지르기 방법위반': 3,
                 '정비불량 제차의 운전금지위반': 3,
                 '중앙선 침범': 14,
                 '직진 및 우회전차의 통행방해': 3,
                 '진로양보 의무 불이행': 3,
                 '차로위반(진로변경 위반)': 3,
                 '통행우선 순위위반': 3},

        '도로형태_대분류': {'고가도로위': 3,
                     '교차로': 2,
                     '기타': 3,
                     '기타/불명': 3,
                     '단일로': 4,
                     '불명': 3,
                     '주차장': 3,
                     '지하도로내': 3},

        '도로형태': {'고가도로위': 6,
                 '교량위': 6,
                 '교차로내': 3,
                 '교차로부근': 4,
                 '교차로횡단보도내': 3,
                 '기타': 6,
                 '기타/불명': 6,
                 '기타단일로': 8,
                 '불명': 6,
                 '지하도로내': 6,
                 '지하차도(도로)내': 6,
                 '터널안': 6,
                 '횡단보도부근': 8,
                 '횡단보도상': 8},

        '당사자종별_1당_대분류': {'개인형이동수단(PM)': 12,
                         '건설기계': 1,
                         '농기계': 2,
                         '불명': 3,
                         '사륜오토바이(ATV)': 12,
                         '승용차': 5,
                         '승합차': 6,
                         '원동기장치자전거': 7,
                         '이륜차': 8,
                         '자전거': 9,
                         '특수차': 10,
                         '화물차': 11},

        '당사자종별_2당_대분류': {'건설기계': 0,
                         '농기계': 1,
                         '불명': 14,
                         '사륜오토바이(ATV)': 14,
                         '승용차': 5,
                         '승합차': 6,
                         '원동기장치자전거': 9,
                         '이륜차': 10,
                         '자전거': 11,
                         '특수차': 12,
                         '화물차': 13}}

    transform_dict_car_man = {
        '주야': {'야간': 0, '주간': 1},

        '요일': {'월': 0, '화': 0, '수': 0, '목': 0, '금': 1, '토': 1, '일': 1},

        '사고유형_중분류': {
            '기타': 2,
            '길가장자리구역통행중': 3,
            '보도통행중': 1,
            '차도통행중': 13,
            '횡단중': 17},

        '법규위반': {
            '과속': 1,
            '교차로 통행방법 위반': 3,
            '기타(운전자법규위반)': 3,
            '보행자 보호의무 위반': 4,
            '보행자과실': 3,
            '부당한 회전': 3,
            '서행 및 일시정지위반': 3,
            '신호위반': 8,
            '안전거리 미확보': 3,
            '안전운전 의무 불이행': 10,
            '정비불량 제차의 운전금지위반': 3,
            '중앙선 침범': 14,
            '직진 및 우회전차의 통행방해': 3,
            '차로위반(진로변경 위반)': 3},

        '도로형태_대분류': {'고가도로위': 3,
                     '교차로': 2,
                     '기타': 3,
                     '기타/불명': 3,
                     '단일로': 4,
                     '불명': 3,
                     '주차장': 3,
                     '지하도로내': 3},

        '도로형태': {
            '고가도로위': 6,
            '교량위': 6,
            '교차로내': 3,
            '교차로부근': 4,
            '교차로횡단보도내': 3,
            '기타': 6,
            '기타/불명': 6,
            '기타단일로': 8,
            '주차장': 6,
            '지하도로내': 6,
            '터널안': 6,
            '횡단보도부근': 8,
            '횡단보도상': 8},

        '당사자종별_1당_대분류': {
            '건설기계': 1,
            '농기계': 2,
            '불명': 3,
            '승용차': 5,
            '승합차': 6,
            '원동기장치자전거': 7,
            '이륜차': 8,
            '자전거': 9,
            '특수차': 10,
            '화물차': 11}}
    transform_dict_car_only = {
        '주야': {'야간': 0, '주간': 1},

        '요일': {'월': 0, '화': 0, '수': 0, '목': 0, '금': 1, '토': 1, '일': 1},

        '사고유형_중분류': {'공작물충돌': 1,
                     '기타': 2,
                     '도로이탈': 4,
                     '전도': 3,
                     '전도전복': 5,
                     '전복': 2,
                     '주/정차차량 충돌': 2},

        '법규위반': {'과속': 1,
                 '교차로 통행방법 위반': 3,
                 '기타(운전자법규위반)': 3,
                 '신호위반': 3,
                 '안전거리 미확보': 9,
                 '안전운전 의무 불이행': 10,
                 '앞지르기 금지위반': 3,
                 '앞지르기 방법위반': 3,
                 '정비불량 제차의 운전금지위반': 3,
                 '중앙선 침범': 14},

        '도로형태_대분류': {'고가도로위': 3,
                     '교차로': 2,
                     '기타': 3,
                     '기타/불명': 3,
                     '단일로': 4,
                     '불명': 3,
                     '주차장': 3,
                     '지하도로내': 3},

        '도로형태': {'고가도로위': 6,
                 '교량위': 0,
                 '교차로내': 3,
                 '교차로부근': 4,
                 '기타': 6,
                 '기타/불명': 6,
                 '기타단일로': 8,
                 '불명': 6,
                 '주차장': 6,
                 '지하도로내': 6,
                 '지하차도(도로)내': 6,
                 '터널안': 6,
                 '횡단보도부근': 8,
                 '횡단보도상': 8}
        ,
        '당사자종별_1당_대분류': {'개인형이동수단(PM)': 12,
                         '건설기계': 11,
                         '농기계': 2,
                         '불명': 12,
                         '사륜오토바이(ATV)': 12,
                         '승용차': 5,
                         '승합차': 6,
                         '원동기장치자전거': 7,
                         '이륜차': 8,
                         '자전거': 9,
                         '특수차': 12,
                         '화물차': 11,
                         '기타': 12}}

    if opt == 0:
        return transfrom_dict_car_car
    elif opt == 1:
        return transform_dict_car_man
    elif opt == 2:
        return transform_dict_car_only
    else:
        return False


def if_trivial(target_list, whole_row, pred_dict, acc_dict, opt):
    avail_opt = ['차대차', '차대사람', '차량단독']

    for i in target_list:
        if i == '사고유형_대분류':
            target_list.remove(i)
            whole_row[9] = avail_opt[opt]
            pred_dict['사고유형_대분류'] = avail_opt[opt]
            acc_dict['사고유형_대분류'] = 1

        if i == '당사자종별_2당_대분류':
            if opt == 1:
                target_list.remove(i)
                whole_row[15] = '보행자'
                pred_dict['당사자종별_2당_대분류'] = '보행자'
                acc_dict['당사자종별_2당_대분류'] = 1

            if opt == 2:
                target_list.remove(i)
                whole_row[15] = '없음'
                pred_dict['당사자종별_2당_대분류'] = '없음'
                acc_dict['당사자종별_2당_대분류'] = 1


def get_opt(whole_row):
    if (whole_row[9] == '차량단독') or (whole_row[15] == '없음'):
        opt = 2

    elif (whole_row[9] == '차대사람') or (whole_row[15] == '보행자'):
        opt = 1
    else:
        opt = 0
    return opt


def test_process_with_option(df_row, target_list, opt):
    avail_opt = ['차대차', '차대사람', '차량단독']
    region_list = ['발생지시도', '발생지시군구']

    ['주야', '요일', '사망자수', '중상자수', '경상자수',
     '부상신고자수', '발생지시도', '발생지시군구', '사고유형_대분류',
     '사고유형_중분류', '법규위반', '도로'
                         '형태_대분류', '도로형태', '당사자종별_1당_대분류', '당사자종별_2당_대분류']

    df_1 = df_row.drop(columns=['사상자수'])
    if not opt == 0:
        df_2 = df_1.drop(columns=['사고유형_대분류', '당사자종별_2당_대분류'])

    else:
        df_2 = df_1.drop(columns=['사고유형_대분류'])

    dic = transfrom_dict(opt)
    if not dic:
        return False
    loc_dic = region_roadtype()

    df_3 = df_2.replace(dic)
    df_4 = df_3.replace(loc_dic)

    drop_list = target_list+['발생지시도']

    drop_list = list(set(drop_list))
    X_test = df_4.drop(columns=drop_list)

    return X_test

def process_with_option(df_row, target_list, opt):
    avail_opt = ['차대차', '차대사람', '차량단독']

    region_list = ['발생지시도', '발생지시군구']
    raw_data = pd.read_csv('./dataset/Kor_Train_traffic_accident_fatalities.csv',
                           encoding="cp949")

    df_1 = raw_data.drop(columns=['발생년', '발생년월일시', '발생분', '사고유형', '사상자수',
                                  '법규위반_대분류', '당사자종별_1당', '당사자종별_2당',
                                  '발생위치X_UTMK', '발생위치Y_UTMK', '경도', '위도'])

    df_2 = df_1.loc[df_1['사고유형_대분류'] == avail_opt[opt]]
    # df_2 : option별 training dataset
    if not opt == 0:
        df_2 = df_2.drop(columns=['사고유형_대분류', '당사자종별_2당_대분류'])

    else:
        df_2 = df_2.drop(columns=['사고유형_대분류'])

    dic = transfrom_dict(opt)
    if not dic:
        return False
    loc_dic = region_roadtype()

    
    df_3 = df_2.replace(dic)
    df_4 = df_3.replace(loc_dic)
    
    X_test = test_process_with_option(df_row, target_list, opt)
    drop_list = target_list+['발생지시도']
    if '발생지시군구'in target_list:
        target_list.remove("발생지시군구")
    drop_list = list(set(drop_list))
    X_train = df_4.drop(columns=drop_list)

    X_concat = pd.concat([X_train, X_test])

    cat = ['주야', '요일', '사고유형_중분류', '법규위반',
           '도로형태_대분류', '도로형태', '당사자종별_1당_대분류',
           '당사자종별_2당_대분류']

    X_features = []
    for idx in list(X_concat):
        if idx in cat:
            X_features.append(True)
        else:
            X_features.append(False)

    X_ohe = OneHotEncoder(categorical_features=X_features)

    dX_concat = X_ohe.fit_transform(X_concat).toarray()
    ddX_concat = pd.DataFrame(dX_concat)

    ddX_test = ddX_concat[0:1]
    ddX_train = ddX_concat[1:]

    y_train_dict = {}

    for i in target_list:
        y_train = df_3[[i]]
        y_features = []

        if i in cat:
            y_features.append(True)
        else:
            y_features.append(False)

        y_ohe = OneHotEncoder(categorical_features=y_features)

        if any(y_features):
            dy_train = y_ohe.fit_transform(y_train).toarray()
        else:
            dy_train = np.array(y_train)
        ddy_train = pd.DataFrame(dy_train)

        y_train_dict[i] = ddy_train

    return ddX_train, y_train_dict, ddX_test


def y_inv_get(target, idx, opt):
    avail_opt = ['차대차', '차대사람', '차량단독']

    region_list = ['발생지시도', '발생지시군구']
    raw_data = pd.read_csv('./dataset/Kor_Train_traffic_accident_fatalities.csv',
                           encoding="cp949")

    df_1 = raw_data.drop(columns=['발생년', '발생년월일시', '발생분', '사고유형',
                                  '법규위반_대분류', '당사자종별_1당', '당사자종별_2당',
                                  '발생위치X_UTMK', '발생위치Y_UTMK', '경도', '위도'])

    df_2 = df_1.loc[df_1['사고유형_대분류'] == avail_opt[opt]]

    if not opt == 0:
        df_2 = df_2.drop(columns=['사고유형_대분류', '당사자종별_2당_대분류'])

    else:
        df_2 = df_2.drop(columns=['사고유형_대분류'])
    df_3 = df_2.reset_index()

    if target == '요일':
        weekday = df_3.loc[idx][target]
        if weekday in ['월', '화', '수', '목']:
            ans = '월'
        else:
            ans = '금'
    elif target == '발생지시도':
        ans = '경기'
    elif target == '발생지시군구':
        ans = '북구'
    else:
        ans = df_3.loc[idx][target]

    return ans

def dense_batch_relu(x, hidden_units, phase, scope):
    with tf.variable_scope(scope):
        # initializer: W: xavier, b: zero
        h1 = tf.contrib.layers.fully_connected(x, hidden_units,
                                               activation_fn=None,
                                               scope='dense')
        h2 = tf.contrib.layers.batch_norm(h1,
                                          center=True, scale=True,
                                          is_training=phase,
                                          scope='bn')
        return tf.nn.relu(h2, 'relu')


def build_neural_network(X_shape, y_shape, hidden_layer_num = 1, hidden_units=10, network_opt=0):
    tf.reset_default_graph()
    inputs = tf.placeholder(tf.float32, shape=[None, X_shape])  # X
    labels = tf.placeholder(tf.float32, shape=[None, y_shape])  # Y
    learning_rate = tf.placeholder(tf.float32)
    # check whether training set or validation - used while batch normalization
    is_training = tf.Variable(True, dtype=tf.bool, name='is_training')

    
    h=[]
    h.append(dense_batch_relu(inputs, hidden_units, is_training, 'layer1'))
    
    for i in range(hidden_layer_num):
        h_next=dense_batch_relu(h[-1], hidden_units, is_training, 'layer'+str(i+2))
        h.append(h_next)
    
    logits = tf.layers.dense(h[-1], y_shape, name='logits')
    # logits = tf.nn.relu(logits) net_opt = 1이면 하면 안되는데 0일때 꼭 해야하면 if문으로

    with tf.name_scope('accuracy'):
        if network_opt == 0:
            #predicted = (tf.nn.tanh(logits) + 1) / 2
            # predicted = tf.nn.sigmoid(logits)
            predicted = tf.maximum(logits, 0)
            pred = 0
            correct_pred = tf.losses.mean_squared_error(labels=labels, predictions=predicted)
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
            
        elif network_opt == 1:
            predicted = tf.maximum(logits, 1)
            pred = 0
            correct_pred = tf.losses.mean_squared_error(labels=labels, predictions=predicted)
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
            
        elif network_opt == 2:
            # select between tanh & sigmoid
            predicted = (tf.nn.tanh(logits) + 1) / 2
            # predicted = tf.nn.sigmoid(logits)
            pred = tf.argmax(predicted, 1)
            correct_pred = tf.equal(tf.argmax(labels, 1), tf.argmax(predicted, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    with tf.name_scope('cost'):
        if network_opt in [0, 1]:
            cost = tf.reduce_mean(-tf.exp(-tf.losses.mean_squared_error(labels=labels, predictions=predicted) + 1e-10))

        elif network_opt == 2:
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=labels, logits=logits)
            cost = tf.reduce_mean(cross_entropy)

    # due to batch_normalization
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # Export the nodes
    export_nodes = ['inputs', 'labels', 'learning_rate', 'is_training', 'logits',
                    'cost', 'optimizer', 'predicted', 'accuracy', 'pred']
    Graph = namedtuple('Graph', export_nodes)
    local_dict = locals()
    graph = Graph(*[local_dict[each] for each in export_nodes])

    return graph


def get_batch(data_x, data_y, batch_size=32):
    batch_n = len(data_x) // batch_size
    for i in range(batch_n):
        batch_x = data_x[i * batch_size:(i + 1) * batch_size]
        batch_y = data_y[i * batch_size:(i + 1) * batch_size]

        yield batch_x, batch_y


def train_network(X_training, X_valid, y_training, y_valid, X_test, model, epochs=5, learning_rate_value=0.0001,
                  batch_size=128, train_collect=50, network_opt=0):
    train_print = train_collect * 2

    x_collect = []
    train_loss_collect = []
    train_acc_collect = []
    valid_loss_collect = []
    valid_acc_collect = []

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        iteration = 0
        for e in range(epochs):
            for batch_x, batch_y in get_batch(X_training, y_training, batch_size):
                iteration += 1
                feed = {model.inputs: X_training,
                        model.labels: y_training,
                        model.learning_rate: learning_rate_value,
                        model.is_training: True
                        }

                train_loss, _, train_acc = sess.run([model.cost, model.optimizer, model.accuracy], feed_dict=feed)
                if iteration % train_collect == 0:
                    x_collect.append(e)
                    train_loss_collect.append(train_loss)
                    train_acc_collect.append(train_acc)

                    """if iteration % train_print==0:
                         print("Epoch: {}/{}".format(e + 1, epochs),
                          "Train Loss: {:.4f}".format(train_loss),
                          "Train Acc: {:.4f}".format(train_acc))"""

                    feed = {model.inputs: X_valid,
                            model.labels: y_valid,
                            model.is_training: False
                            }
                    val_loss, val_acc = sess.run([model.cost, model.accuracy], feed_dict=feed)
                    valid_loss_collect.append(val_loss)
                    valid_acc_collect.append(val_acc)

                    if iteration % train_print == 0:
                        """print("Epoch: {}/{}".format(e + 1, epochs),
                          "Validation Loss: {:.4f}".format(val_loss),
                          "Validation Acc: {:.4f}".format(val_acc))"""

        """print("Optimization Done.")
        plt.plot(x_collect, valid_loss_collect)
        plt.plot(x_collect, valid_acc_collect)
        plt.show()"""
        if network_opt in [0, 1]:
            pred = sess.run([model.predicted], feed_dict={model.inputs: X_test, model.is_training: False})
            return val_acc, pred[0][0][0]

        elif network_opt == 2:
            pred = sess.run([model.pred], feed_dict={model.inputs: X_test, model.is_training: False})
            return val_acc, pred[0][0]


def calculate(X_train, y_train, X_test, num = 1, units = 20, epoches = 40, network_opt=0):
    frac = 0.2
    X_training, X_valid, y_training, y_valid = train_test_split(X_train, y_train, test_size=frac, random_state=43)

    X_shape = X_training.shape[1]
    y_shape = y_training.shape[1]
    model = build_neural_network(X_shape=X_shape, y_shape=y_shape,
     hidden_layer_num = num, hidden_units = units, network_opt=network_opt)
    val_acc, pred_val = train_network(X_training, X_valid, y_training, y_valid, X_test, model, epochs=epoches,
                                 network_opt=network_opt)

    if network_opt in [0, 1]:
        return val_acc, pred_val    

    idx = 0
    while 1:
        if y_train.loc[idx][pred_val]:#.item():
            break
        idx += 1

    if network_opt == 2:
        return val_acc, idx


def target_to_layerinfo(target, opt):
    avail_opt = ['차대차', '차대사람', '차량단독']
    # num: number of layer
    # units: size of each layer
    num = 1
    units = 25
    epoches = 40

    if target in ['사고유형_중분류']:
        epoches = 50
        units = 25
        num = 1
        
    elif target in ['법규위반']:
        epoches = 50
        units = 25
        num = 2
        
    elif target in ['사망자수', '사상자수', '중상자수', '경상자수','부상신고자수']:
        epoches = 100
        units = 100
        num = 2
        
    return num, units, epoches



def preprocess(target_list, whole_row):
    target_dic = {'주야': 'AM/PM', '요일': 'Day', '사망자수': 'Deadpp', '사상자수': 'Totalpp',
                  '중상자수': 'Bhurtpp', '경상자수': 'Shurtpp', '부상신고자수': 'Churtpp',
                  '발생지시도': 'State', '발생지시군구': 'City', '사고유형_대분류': 'Btype',
                  '사고유형_중분류': 'Mtype', '법규위반': 'Law', '도로형태_대분류': 'Broad',
                  '도로형태': 'Road', '당사자종별_1당_대분류': 'B1people', '당사자종별_2당_대분류': 'B2people'}

    avail_target = list(target_dic.keys())
    pred_dict = dict()
    acc_dict = dict()
    opt = get_opt(whole_row)
    if_trivial(target_list, whole_row, pred_dict, acc_dict, opt)

    if '사상자수' in target_list:
        target_list.remove('사상자수')

    if '발생지시도' in target_list:
        target_list.remove('발생지시도')
        
    #if '발생지시군구' in target_list:
    #    target_list.remove('발생지시군구')


    df_row = pd.DataFrame(whole_row).transpose()
    X_train, y_train_dict, X_test = process_with_option(df_row, target_list, opt)

    for target in target_list:
        if target in ['사상자수', '중상자수', '경상자수', '부상신고자수']:
            network_opt = 0
        elif target == '사망자수':
            network_opt = 1
        else:
            network_opt = 2
        num, units, epoches = target_to_layerinfo(target, opt)
        accuracy, idx = calculate(X_train, y_train_dict[target],X_test,
                             num = num, units = units, epoches = epoches, 
                             network_opt=network_opt)

        if network_opt in [0, 1]:
        	result = idx

        elif network_opt == 2:
	        result = y_inv_get(target,idx,opt)
            
            
        pred_dict[target] = result
        acc_dict[target] = accuracy            

    return pred_dict, acc_dict


def fill_row(whole_row, pred_dict):
    for i in pred_dict.keys():
        whole_row[i] = pred_dict[i]


def fill_blank(result_data, idx, pred_dict):
    target_dic = {'A': '주야', 'B': '요일', 'C': '사망자수', 'D': '사상자수',
                  'E': '중상자수', 'F': '경상자수', 'G': '부상신고자수',
                  'H': '발생지시도', 'I': '발생지시군구', 'J': '사고유형_대분류',
                  'K': '사고유형_중분류', 'L': '법규위반', 'M': '도로형태_대분류',
                  'N': '도로형태', 'O': '당사자종별_1당_대분류',
                  'P': '당사자종별_2당_대분류'}

    result_len = result_data.shape[0]

    for i in range(result_len):
        if result_data.loc[i][0] == (idx + 2):
            row = result_data.loc[i][1]
            target = target_dic[row]
            ans = pred_dict[target]
                
            #rint("target: ",target, "// answer: ", ans)
            result_data.at[i, '값'] = ans


def main():
    #num, units, epoches = target_to_layerinfo('사망자수', 1)
    #print('num: ' ,num,'units: ',units, 'eopches: ',epoches)

    test_data = pd.read_csv('./dataset/test_kor.csv', encoding="cp949")
    result_data = pd.read_csv('./dataset/result_kor.csv', encoding="cp949")
    result_data['값'] = result_data['값'].astype(str)

    for index, row in test_data.iterrows():

        if True:#index >= 30 and index < 2001:
            print('idx: ', index)
            target_list = []
            for i in range(len(row)):
                if row.isnull()[i]:
                    target_list.append(test_data.keys()[i])
            pred_dict, acc_dict = preprocess(target_list, row)

            fill_row(row, pred_dict)
            
            #print('row: ',row)
            #print('pred_dict: ',pred_dict)


            if np.isnan(row['사상자수']):
                people = 0
                for j in ['사망자수', '중상자수', '경상자수', '부상신고자수']:
                    people += row[j]
                row['사상자수'] = people
                pred_dict['사상자수'] = people
                
            if type(row['발생지시도']) != str:
                if np.isnan(row['발생지시도']):
                    row['발생지시도'] = '경기'
                    pred_dict['발생지시도'] = '경기'
                    acc_dict['발생지시도'] = 0

            if type(row['발생지시군구']) != str:
                if np.isnan(row['발생지시군구']):
                    row['발생지시군구'] = '북구'
                    pred_dict['발생지시군구'] = '북구'
                    acc_dict['발생지시군구'] = 0

         
                
            #print('row: ',row)
            #print('pred_dict: ',pred_dict)
            fill_blank(result_data, index, pred_dict)

            for target in pred_dict:
                if target in ['사망자수', '사상자수', '중상자수', '경상자수', '부상신고자수']:
                    print(target, '//', pred_dict[target])

                else:
                    print(target, '//', pred_dict[target], '// acc: ', acc_dict[target])
                    
                    
    print("Done")
    result_data.to_csv("./result/result_kor.csv", index=False, encoding="cp949")


main()