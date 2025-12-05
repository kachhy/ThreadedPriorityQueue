#ifndef COMMUNICATION_QUEUE_H
#define COMMUNICATION_QUEUE_H

#include <condition_variable>
#include <optional>
#include <vector>
#include <thread>
#include <mutex>

// We default to a minheap.
struct LessThanComparitor {
    template <typename T>
    bool operator()(const T& a, const T& b) const {
        return a < b;
    }
};

template <typename T, typename Comp = LessThanComparitor>
class ThreadedPriorityQueue {
    std::vector<T> m_heapVector;
    std::condition_variable m_readCondition;
    mutable std::mutex m_commMutex;
    bool m_isDone = false; // Controls when to resume thread condition

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
        const size_t n = m_heapVector.size();

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

    // Push and pop
    inline void push(const T& item) noexcept {
        std::lock_guard<std::mutex> lock(m_commMutex);
        m_heapVector.emplace_back(item);
        percolate_up(m_heapVector.size() - 1);
        m_readCondition.notify_one();
    }
    
    inline T pop() {
        std::lock_guard<std::mutex> lock(m_commMutex);
        if (m_heapVector.empty())
            throw std::runtime_error("pop() attempted on empty communication queue.");

        T temp = std::move(m_heapVector.front());
        
        if (m_heapVector.size() > 1) {
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

        m_heapVector.emplace_back(item);
        percolate_up(m_heapVector.size() - 1);
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
        
        if (m_heapVector.size() > 1) {
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
        return m_heapVector.size();
    }

    inline bool empty() const noexcept {
        return !m_heapVector.size();
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

#endif // COMMUNICATION_QUEUE_H