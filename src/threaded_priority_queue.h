#ifndef THREADED_PRIORITY_QUEUE_H
#define THREADED_PRIORITY_QUEUE_H

#include <condition_variable>
#include <type_traits>
#include <optional>
#include <cstring>
#include <thread>
#include <mutex>

// We default to a minheap.
struct MinHeapComparitor {
    template <typename T>
    bool operator()(const T& a, const T& b) const {
        return a < b;
    }
};

struct MaxHeapComparitor {
    template <typename T>
    bool operator()(const T& a, const T& b) const {
        return a > b;
    }
};

template <typename T, typename Comp = MinHeapComparitor>
class ThreadedPriorityQueue {
    struct HeapVec {
        T* m_arr = nullptr;
        size_t m_size = 0, m_capacity = 0;

        HeapVec() = default;
        // Freeing handled by TPQ

        inline void reserve(size_t cap) noexcept {
            if (cap <= m_capacity)
                return;

            T* temp = m_arr;
            m_arr = new T[cap];

            if (temp) {
                if constexpr (std::is_trivially_copyable_v<T>) // Bitwise optimized copy for trivial types
                    memcpy(m_arr, temp, m_size * sizeof(T));
                else
                    for (size_t i = 0; i < m_size; ++i)
                        m_arr[i] = std::move(temp[i]);
                
                delete [] temp;
            }

            m_capacity = cap;
        }

        inline bool empty() const noexcept {
            return !m_size;
        }

        inline T& front() {
            return m_arr[0];
        }

        inline T& back() {
            return m_arr[m_size - 1];
        }

        inline void pop_back() noexcept {
            if (m_size > 0)
                --m_size;
        }

        template <typename... Args>
        inline void emplace_back(Args&&... args) noexcept {
            if (m_size >= m_capacity)
                reserve((m_capacity == 0) ? 1 : m_capacity * 2);
            
            new (m_arr + m_size++) T(std::forward<Args>(args)...); // Construct in-place
        }

        inline void push_back(T&& element) noexcept {
            if (m_size >= m_capacity)
                reserve((m_capacity == 0) ? 1 : m_capacity * 2);

            m_arr[m_size++] = std::move(element);
        }

        inline void push_back(const T& element) noexcept {
            if (m_size >= m_capacity)
                reserve((m_capacity == 0) ? 1 : m_capacity * 2);

            m_arr[m_size++] = element;
        }

        inline const T& operator[](const size_t i) const noexcept {
            return m_arr[i];
        }

        inline T& operator[](const size_t i) noexcept {
            return m_arr[i];
        }
    };

    // Private heap variables
    HeapVec m_heapVector;
    std::condition_variable m_readCondition;
    mutable std::mutex m_commMutex;
    bool m_isDone = false;

    // Private heap functions
    inline void percolate_up(size_t index) noexcept {
        if (!index)
            return;
        
        size_t parent_index = (index - 1) / 2;

        while (index > 0 && Comp{}(m_heapVector[index], m_heapVector[parent_index])) {
            std::swap(m_heapVector[index], m_heapVector[parent_index]);
            index = parent_index;

            if (index > 0)
                parent_index = (index - 1) / 2;
        }
    }

    inline void percolate_down(size_t index) noexcept {
        const size_t n = m_heapVector.m_size;

        while (2 * index + 1 < n) {
            const size_t left_child = 2 * index + 1;
            const size_t right_child = 2 * index + 2;
            
            size_t largest_index = index;
            if (left_child < n
                && Comp{}(m_heapVector[left_child], m_heapVector[largest_index]))
                largest_index = left_child;
                
            if (right_child < n
                && Comp{}(m_heapVector[right_child], m_heapVector[largest_index]))
                largest_index = right_child;
            
            if (largest_index == index)
                break;

            std::swap(m_heapVector[index], m_heapVector[largest_index]);
            index = largest_index;
        }
    }
public:
    ThreadedPriorityQueue() = default;
    ThreadedPriorityQueue(const size_t reserve) { m_heapVector.reserve(reserve); }

    // Disable copying and moving to prevent double-free issues due to raw pointer management
    ThreadedPriorityQueue(const ThreadedPriorityQueue&) = delete;
    ThreadedPriorityQueue& operator=(const ThreadedPriorityQueue&) = delete;
    ThreadedPriorityQueue(ThreadedPriorityQueue&&) = delete;
    ThreadedPriorityQueue& operator=(ThreadedPriorityQueue&&) = delete;

    ~ThreadedPriorityQueue() {
        if (m_heapVector.m_arr)
            delete [] m_heapVector.m_arr;
    }

    // Push and pop
    inline void push(const T& item) noexcept {
        std::lock_guard<std::mutex> lock(m_commMutex);
        m_heapVector.push_back(item);
        percolate_up(m_heapVector.m_size - 1);
        m_readCondition.notify_one();
    }

    inline void push(T&& item) noexcept {
        std::lock_guard<std::mutex> lock(m_commMutex);
        m_heapVector.push_back(std::move(item));
        percolate_up(m_heapVector.m_size - 1);
        m_readCondition.notify_one();
    }

    template <typename... Args>
    inline void push(Args&&... args) noexcept {
        std::lock_guard<std::mutex> lock(m_commMutex);
        m_heapVector.emplace_back(std::forward<Args>(args)...);
        percolate_up(m_heapVector.m_size - 1);
        m_readCondition.notify_one();
    }
    
    inline T pop() {
        std::lock_guard<std::mutex> lock(m_commMutex);
        if (m_heapVector.empty())
            throw std::runtime_error("pop() attempted on empty communication queue.");

        T temp = std::move(m_heapVector.front());
        
        if (m_heapVector.m_size > 1) {
            m_heapVector[0] = std::move(m_heapVector.back());
            m_heapVector.pop_back();
            percolate_down(0);
        } else
            m_heapVector.pop_back();
        
        // Notify if the queue became empty, as this state is used by wait_empty_push
        if (m_heapVector.empty())
            m_readCondition.notify_one();
        
        return temp;
    }

    // Threaded push/pop
    inline void wait_empty_push(const T& item) { // Waits til empty
        std::unique_lock<std::mutex> lock(m_commMutex);
        
        // Wait until empty or done
        m_readCondition.wait(lock, [this] {
            return m_heapVector.empty() || m_isDone;
        });

        if (m_isDone)
            return;

        m_heapVector.push_back(item);
        percolate_up(m_heapVector.m_size - 1);
        m_readCondition.notify_one();
    }

    inline void wait_empty_push(T&& item) { // Waits til empty
        std::unique_lock<std::mutex> lock(m_commMutex);
        
        // Wait until empty or done
        m_readCondition.wait(lock, [this] {
            return m_heapVector.empty() || m_isDone;
        });

        if (m_isDone)
            return;

        m_heapVector.push_back(std::move(item));
        percolate_up(m_heapVector.m_size - 1);
        m_readCondition.notify_one();
    }

    template <typename... Args>
    inline void wait_empty_emplace(Args&&... args) { // Waits til empty
        std::unique_lock<std::mutex> lock(m_commMutex);
        
        // Wait until empty or done
        m_readCondition.wait(lock, [this] {
            return m_heapVector.empty() || m_isDone;
        });

        if (m_isDone)
            return;

        m_heapVector.emplace_back(std::forward<Args>(args)...);
        percolate_up(m_heapVector.m_size - 1);
        m_readCondition.notify_one();
    }
    
    inline std::optional<T> wait_nonempty_pop() { // Waits til non-empty
        std::unique_lock<std::mutex> lock(m_commMutex);
        
        // Wait until non-empty or done
        m_readCondition.wait(lock, [this] {
            return !m_heapVector.empty() || m_isDone;
        });

        if (m_heapVector.empty())
            return std::nullopt;

        T temp = std::move(m_heapVector.front());
        
        if (m_heapVector.m_size > 1) {
            m_heapVector[0] = std::move(m_heapVector.back());
            m_heapVector.pop_back();
            percolate_down(0);
        } else
            m_heapVector.pop_back();
        
        // Notify if the queue became empty, as this state is used by wait_empty_push
        if (m_heapVector.empty())
            m_readCondition.notify_one();

        return std::make_optional<T>(std::move(temp));
    }

    // Strict getters
    inline const T& top() const {
        std::lock_guard<std::mutex> lock(m_commMutex);
        if (m_heapVector.empty())
            throw std::runtime_error("top() attempted on empty communication queue.");
        return m_heapVector.front();
    }

    inline size_t size() const noexcept {
        return m_heapVector.m_size;
    }

    inline bool empty() const noexcept {
        return m_heapVector.empty();
    }

    // Threaded getters
    inline std::optional<T> wait_and_get_top() const {
        std::unique_lock<std::mutex> lock(m_commMutex);
        m_readCondition.wait(lock, [this] {
            return !m_heapVector.empty() || m_isDone;
        });

        if (m_heapVector.empty())
            return std::nullopt;

        return std::make_optional<T>(m_heapVector.front());
    }

    // Done function
    inline void done() noexcept {
        {
            std::lock_guard<std::mutex> lock(m_commMutex);
            m_isDone = true;
        }

        // Notify after unlock
        m_readCondition.notify_all();
    }

    inline bool is_done() const noexcept {
        return m_isDone;
    }
};

#endif // THREADED_PRIORITY_QUEUE_H