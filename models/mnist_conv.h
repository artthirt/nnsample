#ifndef MIST_CONV_H
#define MIST_CONV_H

#include "custom_types.h"
#include "mnist_reader.h"
#include "nn.h"

#include "convnn.h"

#include <random>

class mnist_conv
{
public:
	mnist_conv();
	/**
	 * @brief forward
	 * @param X
	 * @return
	 */
	ct::Matf forward(const ct::Matf& X);
	/**
	 * @brief forward
	 * @param index
	 * @param count
	 * @param use_gpu
	 * @return
	 */
	ct::Matf forward(int index, int count);
	/**
	 * @brief forward_test
	 * @param index
	 * @param count
	 * @param use_gpu
	 * @return
	 */
	ct::Matf forward_test(int index, int count);
	/**
	 * @brief setAlpha
	 * @param alpha
	 */
	void setAlpha(double alpha);
	/**
	 * @brief setLayers
	 * @param layers
	 */
	void setLayers(const std::vector<int>& layers);
	/**
	 * @brief iteration
	 * @return
	 */
	uint iteration() const;
	/**
	 * @brief getEstimate
	 * @param batch
	 * @param l2
	 * @param accuracy
	 * @param use_gpu
	 */
	void getEstimate(int batch, double &l2, double &accuracy);
	/**
	 * @brief getEstimateTest
	 * @param batch
	 * @param l2
	 * @param accuracy
	 * @param use_gpu
	 */
	void getEstimateTest(int batch, double &l2, double &accuracy);
	/**
	 * @brief init
	 * @param seed
	 */
	void init(int seed);
	/**
	 * @brief pass_batch
	 * @param batch
	 */
	void pass_batch(int batch);
	/**
	 * @brief setMnist
	 * @param mnist
	 */
	void setMnist(mnist_reader* mnist);
	void setConvLength(const std::vector< int > &count_cnvW);

	void random_update_weights();

	std::vector< std::vector< convnn::convnn<float> > > &cnv();

	std::vector<std::vector<ct::Matf> > cnvW();

private:
	std::vector< int > m_layers;
	std::vector< std::vector< convnn::convnn<float> > > m_cnv;
	std::vector< int > m_count_cnvW;
	std::vector< ct::Matf > m_W;
	std::vector< ct::Matf > m_b;
	std::vector< ct::Matf > m_prevW;
	std::vector< ct::Matf > m_prevb;
	mnist_reader *m_mnist;
	std::mt19937 m_generator;
	uint m_iteration;
	int m_conv_length;

	ct::Size m_cnv_out_size;
	int m_cnv_out_len;

	nn::AdamOptimizer<float> m_AdamOptimizer;

	void pass_batch(const ct::Matf& X, const ct::Matf& y);

	void getX(ct::Matf& X, int batch);
	void getXyTest(ct::Matf &X, ct::Matf &yp, int batch);
	void getXy(ct::Matf& X, ct::Matf& y, int batch);
	void randX(ct::Matf& X);
	void getBatchIds(std::vector< int >& indexes, int batch = -1);

	void conv(const ct::Matf &X, ct::Matf &X_out);

	void save_weights();
	void restore_weights();

};

#endif // MIST_CONV_H
