#ifndef SHARED_MEMORY_H
#define SHARED_MEMORY_H

#include <vector>
#include <iostream>
#include <sstream>
#include <string>

namespace sm{

template< class T >
class shared_memory
{
public:
	shared_memory(){
		m_val = 0;
		m_ref = 0;
	}
	shared_memory(const shared_memory<T>& val){
		m_val = val.m_val;
		m_ref = val.m_ref;
		if(m_ref)
			(*m_ref)++;
	}
	~shared_memory(){
		if(m_ref){
			(*m_ref)--;
			if(*m_ref <= 0 && m_val){
				delete m_val;
				delete m_ref;
			}
		}
	}

	shared_memory<T>& operator=(const shared_memory<T>& val){
		if(m_val && m_ref){
			--(*m_ref);
			if(*m_ref <= 0 && m_val != val.m_val){
				delete m_val;
				delete m_ref;
			}
			if(m_val == val.m_val){
				//(*m_ref)++;
			}else{
				m_ref = 0;
			}
		}

		m_val = val.m_val;
		m_ref = val.m_ref;
		if(m_ref)
			(*m_ref)++;
		return *this;
	}

	bool empty() const{
		return m_val == 0;
	}
	T* get(){
		return m_val;
	}
	T* get() const{
		return m_val;
	}
	T& operator*(){
		return *m_val;
	}
	int ref() const{
		if(m_ref)
			return *m_ref;
		return -1;
	}

	template< class N >
	friend shared_memory< N > make_shared();

private:
	T* m_val;
	int *m_ref;

	shared_memory(T* val){
		m_val = 0;
		m_ref = 0;
		if(val){
			m_ref = new int(1);
			m_val = val;
		}
	}
};

template< class T >
shared_memory< T > make_shared()
{
	shared_memory< T > res(new T());

	return res;
}

}

#endif // SHARED_MEMORY_H
