import tensorflow as tf

class SkipGANomaly(object):

    def __init__(self, height, width, channel, z_dim, w_enc=1, w_con=50, w_adv=1, leaning_rate=1e-3):

        print("\nInitializing Neural Network...")
        self.height, self.width, self.channel = height, width, channel
        self.k_size, self.z_dim = 3, z_dim
        self.w_enc, self.w_con, self.w_adv = w_enc, w_con, w_adv
        self.leaning_rate = leaning_rate

        self.x = tf.compat.v1.placeholder(tf.float32, [None, self.height, self.width, self.channel])
        self.batch_size = tf.placeholder(tf.int32, shape=[])

        self.weights, self.biasis = [], []
        self.w_names, self.b_names = [], []
        self.fc_shapes, self.conv_shapes = [], []
        self.features_real, self.features_fake = [], []

        self.z_code, self.x_hat, self.z_code_hat, self.dis_x, self.dis_x_hat, self.features_real, self.features_fake =\
            self.build_model(input=self.x, ksize=self.k_size)

        # Loss 1: Encoding loss (L2 distance)
        self.loss_enc = tf.compat.v1.reduce_sum(tf.square(self.z_code - self.z_code_hat), axis=(1))
        # Loss 2: Restoration loss (L1 distance)
        self.loss_con = tf.compat.v1.reduce_sum(tf.abs(self.x - self.x_hat), axis=(1, 2, 3))
        # Loss 3: Adversarial loss (L2 distance)
        self.loss_adv = tf.compat.v1.reduce_sum(tf.square(self.dis_x - self.dis_x_hat), axis=(1))
        for fidx, _ in enumerate(self.features_real):
            feat_dim = len(self.features_real[fidx].shape)
            if(feat_dim == 4):
                self.loss_adv += tf.compat.v1.reduce_sum(tf.square(self.features_real[fidx] - self.features_fake[fidx]), axis=(1, 2, 3))
            elif(feat_dim == 3):
                self.loss_adv += tf.compat.v1.reduce_sum(tf.square(self.features_real[fidx] - self.features_fake[fidx]), axis=(1, 2))
            elif(feat_dim == 2):
                self.loss_adv += tf.compat.v1.reduce_sum(tf.square(self.features_real[fidx] - self.features_fake[fidx]), axis=(1))
            else:
                self.loss_adv += tf.compat.v1.reduce_sum(tf.square(self.features_real[fidx] - self.features_fake[fidx]))

        self.mean_loss_enc = tf.compat.v1.reduce_mean(self.loss_enc)
        self.mean_loss_con = tf.compat.v1.reduce_mean(self.loss_con)
        self.mean_loss_adv = tf.compat.v1.reduce_mean(self.loss_adv)

        self.loss = tf.compat.v1.reduce_mean((self.w_enc * self.loss_enc) + (self.w_con * self.loss_con) + (self.w_adv * self.loss_adv))

        #default: beta1=0.9, beta2=0.999
        self.optimizer = tf.compat.v1.train.AdamOptimizer( \
            self.leaning_rate, beta1=0.5, beta2=0.999).minimize(self.loss)

        tf.compat.v1.summary.scalar('loss_enc', self.mean_loss_enc)
        tf.compat.v1.summary.scalar('loss_con', self.mean_loss_con)
        tf.compat.v1.summary.scalar('loss_adv', self.mean_loss_adv)
        tf.compat.v1.summary.scalar('loss_tot', self.loss)
        self.summaries = tf.compat.v1.summary.merge_all()

    def build_model(self, input, ksize=3):

        with tf.name_scope('generator') as scope_gen:
            z_code = self.encoder(input=input, ksize=ksize)
            x_hat = self.decoder(input=z_code, ksize=ksize)
            z_code_hat = self.encoder(input=x_hat, ksize=ksize)

        with tf.name_scope('discriminator') as scope_dis:
            dis_x, features_real = self.discriminator(input=input, ksize=ksize)
            dis_x_hat, features_fake = self.discriminator(input=x_hat, ksize=ksize)

        return z_code, x_hat, z_code_hat, dis_x, dis_x_hat, features_real, features_fake

    def encoder(self, input, ksize=3):

        print("\nEncode-1")
        conv1_1 = self.conv2d(input=input, stride=1, padding='SAME', \
            filter_size=[ksize, ksize, 1, 16], activation="lrelu", name="enconv1_1")
        conv1_2 = self.conv2d(input=conv1_1, stride=1, padding='SAME', \
            filter_size=[ksize, ksize, 16, 16], activation="lrelu", name="enconv1_2")
        maxp1 = self.maxpool(input=conv1_2, ksize=2, strides=2, padding='SAME', name="max_pool1")
        self.concat1 = maxp1
        self.conv_shapes.append(conv1_2.shape)

        print("Encode-2")
        conv2_1 = self.conv2d(input=maxp1, stride=1, padding='SAME', \
            filter_size=[ksize, ksize, 16, 32], activation="lrelu", name="enconv2_1")
        conv2_2 = self.conv2d(input=conv2_1, stride=1, padding='SAME', \
            filter_size=[ksize, ksize, 32, 32], activation="lrelu", name="enconv2_2")
        maxp2 = self.maxpool(input=conv2_2, ksize=2, strides=2, padding='SAME', name="max_pool2")
        self.concat2 = maxp2
        self.conv_shapes.append(conv2_2.shape)

        print("Encode-3")
        conv3_1 = self.conv2d(input=maxp2, stride=1, padding='SAME', \
            filter_size=[ksize, ksize, 32, 64], activation="lrelu", name="enconv3_1")
        conv3_2 = self.conv2d(input=conv3_1, stride=1, padding='SAME', \
            filter_size=[ksize, ksize, 64, 64], activation="lrelu", name="enconv3_2")
        self.concat3 = conv3_2
        self.conv_shapes.append(conv3_2.shape)

        print("Dense (Fully-Connected)")
        self.fc_shapes.append(conv3_2.shape)
        [n, h, w, c] = self.fc_shapes[0]
        fulcon_in = tf.compat.v1.reshape(conv3_2, shape=[self.batch_size, h*w*c], name="enfulcon_in")
        fulcon1 = self.fully_connected(input=fulcon_in, num_inputs=int(h*w*c), \
            num_outputs=512, activation="lrelu", name="enfullcon1")

        z_code = self.fully_connected(input=fulcon1, num_inputs=int(fulcon1.shape[1]), \
            num_outputs=self.z_dim, activation="None", name="encode")

        return z_code

    def decoder(self, input, ksize=3):

        print("\nDecode-Dense")
        [n, h, w, c] = self.fc_shapes[0]
        fulcon2 = self.fully_connected(input=input, num_inputs=int(self.z_dim), \
            num_outputs=512, activation="lrelu", name="defullcon2")
        fulcon3 = self.fully_connected(input=fulcon2, num_inputs=int(fulcon2.shape[1]), \
            num_outputs=int(h*w*c), activation="lrelu", name="defullcon3")
        fulcon_out = tf.compat.v1.reshape(fulcon3, shape=[self.batch_size, h, w, c], name="defulcon_out")

        print("Decode-1")
        concated1 = tf.concat([fulcon_out, self.concat3], axis=3)
        convt1_1 = self.conv2d(input=concated1, stride=1, padding='SAME', \
            filter_size=[ksize, ksize, 128, 64], activation="lrelu", name="deconv1_1")
        convt1_2 = self.conv2d(input=convt1_1, stride=1, padding='SAME', \
            filter_size=[ksize, ksize, 64, 64], activation="lrelu", name="deconv1_2")

        print("Decode-2")
        [n, h, w, c] = self.conv_shapes[-2]
        concated2 = tf.concat([convt1_2, self.concat2], axis=3)
        convt2_1 = self.conv2d_transpose(input=concated2, stride=2, padding='SAME', \
            output_shape=[self.batch_size, h, w, c], filter_size=[ksize, ksize, c, 64+32], \
            dilations=[1, 1, 1, 1], activation="lrelu", name="deconv2_1")
        convt2_2 = self.conv2d(input=convt2_1, stride=1, padding='SAME', \
            filter_size=[ksize, ksize, c, 32], activation="lrelu", name="deconv2_2")

        print("Decode-3")
        [n, h, w, c] = self.conv_shapes[-3]
        concated3 = tf.concat([convt2_2, self.concat1], axis=3)
        convt3_1 = self.conv2d_transpose(input=concated3, stride=2, padding='SAME', \
            output_shape=[self.batch_size, h, w, c], filter_size=[ksize, ksize, c, 32+16], \
            dilations=[1, 1, 1, 1], activation="lrelu", name="deconv3_1")
        convt3_2 = self.conv2d(input=convt3_1, stride=1, padding='SAME', \
            filter_size=[ksize, ksize, c, 16], activation="lrelu", name="deconv3_2")
        convt3_3 = self.conv2d(input=convt3_2, stride=1, padding='SAME', \
            filter_size=[ksize, ksize, 16, 1], activation="None", name="deconv3_3")
        convt3_3 = tf.compat.v1.clip_by_value(convt3_3, 1e-12, 1-(1e-12))

        return convt3_3

    def discriminator(self, input, ksize=3):

        featurebank = []

        print("\nDiscriminate-1")
        conv1_1 = self.conv2d(input=input, stride=1, padding='SAME', \
            filter_size=[ksize, ksize, 1, 16], activation="elu", name="disconv1_1")
        featurebank.append(conv1_1)
        conv1_2 = self.conv2d(input=conv1_1, stride=1, padding='SAME', \
            filter_size=[ksize, ksize, 16, 16], activation="elu", name="disconv1_2")
        featurebank.append(conv1_2)
        maxp1 = self.maxpool(input=conv1_2, ksize=2, strides=2, padding='SAME', name="max_pool1")

        print("Discriminate-2")
        conv2_1 = self.conv2d(input=maxp1, stride=1, padding='SAME', \
            filter_size=[ksize, ksize, 16, 32], activation="elu", name="disconv2_1")
        featurebank.append(conv2_1)
        conv2_2 = self.conv2d(input=conv2_1, stride=1, padding='SAME', \
            filter_size=[ksize, ksize, 32, 32], activation="elu", name="disconv2_2")
        featurebank.append(conv2_2)
        maxp2 = self.maxpool(input=conv2_2, ksize=2, strides=2, padding='SAME', name="max_pool2")

        print("Discriminate-3")
        conv3_1 = self.conv2d(input=maxp2, stride=1, padding='SAME', \
            filter_size=[ksize, ksize, 32, 64], activation="elu", name="disconv3_1")
        featurebank.append(conv3_1)
        conv3_2 = self.conv2d(input=conv3_1, stride=1, padding='SAME', \
            filter_size=[ksize, ksize, 64, 64], activation="elu", name="disconv3_2")
        featurebank.append(conv3_2)

        print("Dense (Fully-Connected)")
        [n, h, w, c] = conv3_2.shape
        fulcon_in = tf.compat.v1.reshape(conv3_2, shape=[self.batch_size, h*w*c], name="disfulcon_in")
        fulcon1 = self.fully_connected(input=fulcon_in, num_inputs=int(h*w*c), \
            num_outputs=512, activation="elu", name="disfullcon1")
        featurebank.append(fulcon1)
        disc_score = self.fully_connected(input=fulcon1, num_inputs=int(fulcon1.shape[1]), \
            num_outputs=1, activation="sigmoid", name="disc_sco")
        featurebank.append(disc_score)

        return disc_score, featurebank

    def initializer(self):
        return tf.compat.v1.initializers.variance_scaling(distribution="untruncated_normal", dtype=tf.dtypes.float32)

    def maxpool(self, input, ksize, strides, padding, name=""):

        out_maxp = tf.compat.v1.nn.max_pool(value=input, \
            ksize=ksize, strides=strides, padding=padding, name=name)
        print("Max-Pool", input.shape, "->", out_maxp.shape)

        return out_maxp

    def activation_fn(self, input, activation="relu", name=""):

        if("sigmoid" == activation):
            out = tf.compat.v1.nn.sigmoid(input, name='%s_sigmoid' %(name))
        elif("tanh" == activation):
            out = tf.compat.v1.nn.tanh(input, name='%s_tanh' %(name))
        elif("relu" == activation):
            out = tf.compat.v1.nn.relu(input, name='%s_relu' %(name))
        elif("lrelu" == activation):
            out = tf.compat.v1.nn.leaky_relu(input, name='%s_lrelu' %(name))
        elif("elu" == activation):
            out = tf.compat.v1.nn.elu(input, name='%s_elu' %(name))
        else: out = input

        return out

    def batch_normalization(self, input):

        mean = tf.compat.v1.reduce_mean(input)
        std = tf.compat.v1.math.reduce_std(input)

        return (input - mean) / (std + 1e-12)

    def variable_maker(self, var_bank, name_bank, shape, name=""):

        try:
            var_idx = name_bank.index(name)
        except:
            variable = tf.compat.v1.get_variable(name=name, \
                shape=shape, initializer=self.initializer())

            var_bank.append(variable)
            name_bank.append(name)
        else:
            variable = var_bank[var_idx]

        return var_bank, name_bank, variable

    def conv2d(self, input, stride, padding, \
        filter_size=[3, 3, 16, 32], dilations=[1, 1, 1, 1], activation="relu", name=""):

        self.weights, self.w_names, weight = self.variable_maker(var_bank=self.weights, name_bank=self.w_names, \
            shape=filter_size, name='%s_w' %(name))
        self.biasis, self.b_names, bias = self.variable_maker(var_bank=self.biasis, name_bank=self.b_names, \
            shape=[filter_size[-1]], name='%s_b' %(name))

        out_conv = tf.compat.v1.nn.conv2d(
            input=input,
            filter=weight,
            strides=[1, stride, stride, 1],
            padding=padding,
            use_cudnn_on_gpu=True,
            data_format='NHWC',
            dilations=dilations,
            name='%s_conv' %(name),
        )
        out_bias = tf.math.add(out_conv, bias, name='%s_add' %(name))
        out_bias = self.batch_normalization(input=out_bias)

        print("Conv", input.shape, "->", out_bias.shape)
        return self.activation_fn(input=out_bias, activation=activation, name=name)

    def conv2d_transpose(self, input, stride, padding, output_shape, \
        filter_size=[3, 3, 16, 32], dilations=[1, 1, 1, 1], activation="relu", name=""):

        self.weights, self.w_names, weight = self.variable_maker(var_bank=self.weights, name_bank=self.w_names, \
            shape=filter_size, name='%s_w' %(name))
        self.biasis, self.b_names, bias = self.variable_maker(var_bank=self.biasis, name_bank=self.b_names, \
            shape=[filter_size[-2]], name='%s_b' %(name))

        print(weight.shape, bias.shape)
        out_conv = tf.compat.v1.nn.conv2d_transpose(
            value=input,
            filter=weight,
            output_shape=output_shape,
            strides=[1, stride, stride, 1],
            padding=padding,
            data_format='NHWC',
            dilations=dilations,
            name='%s_conv_tr' %(name),
        )
        print(weight.shape, input.shape, out_conv.shape, bias.shape)
        out_bias = tf.math.add(out_conv, bias, name='%s_add' %(name))
        out_bias = self.batch_normalization(input=out_bias)

        print("Conv-Tr", input.shape, "->", out_bias.shape)
        return self.activation_fn(input=out_bias, activation=activation, name=name)

    def fully_connected(self, input, num_inputs, num_outputs, activation="relu", name=""):

        self.weights, self.w_names, weight = self.variable_maker(var_bank=self.weights, name_bank=self.w_names, \
            shape=[num_inputs, num_outputs], name='%s_w' %(name))
        self.biasis, self.b_names, bias = self.variable_maker(var_bank=self.biasis, name_bank=self.b_names, \
            shape=[num_outputs], name='%s_b' %(name))

        out_mul = tf.compat.v1.matmul(input, weight, name='%s_mul' %(name))
        out_bias = tf.math.add(out_mul, bias, name='%s_add' %(name))
        out_bias = self.batch_normalization(input=out_bias)

        print("Full-Con", input.shape, "->", out_bias.shape)
        return self.activation_fn(input=out_bias, activation=activation, name=name)
