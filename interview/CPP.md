**C语言**

- const的作用有哪些，谈一谈你对const的理解？
  - const是一个C语言的关键字，它的作用是限定一个变量不允许被改变。const是给系统看，让系统不要改变我的值。const也是给程序员看，让程序员看这里为什么要用const，到底能不能改这个值，而不是不管三七二十一的，无视const，用指针调用指针来把const的作用给无视掉。
- **描述char\*、const char\*、char\* const、const char\* const的区别？**
  - const char * 修饰 * ，所以值不能改变。 即 指向常量的指针变量。
  - char * const 修饰 s ，所以指向不可变。 即 指向变量的指针常量。
- 指针常量和常量指针有什么区别？
  - 指针常量时指针类型的常量，常量指针是指向常量的指针。
- **[static](https://www.zhihu.com/search?q=static&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A2590210440})**的作用是什么，什么情况下用到static？**全局变量与局部变量的区别？**宏定义的作用是什么？**内存对齐的概念？为什么会有内存对齐？**[inline](https://www.zhihu.com/search?q=inline&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A2590210440}) **[内联函数](https://www.zhihu.com/search?q=内联函数&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A1868370927})的特点有哪些？它的优缺点是什么？**如何避免[野指针](https://www.zhihu.com/search?q=野指针&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A2590210440})？
  - static : 静态局部变量，不可被其他文件所用，其他文件可定义重名变量。
  - 全局变量的生命周期与程序一致，局部变量的程序周期在函数中，作用域也是。
  - 宏定义：在C++头文件中，我们常常会用到几个宏定义（#ifndef #define #endif）。
  - 内存对齐：https://www.zhihu.com/question/627238873/answer/3260521889
  - 内联函数：
  - 野指针：

- **如何计算结构体长度？**
- sizeof和[strlen](https://www.zhihu.com/search?q=strlen&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A1868370927})有什么区别？
- **知道条件变量吗？条件变量为什么要和锁配合使用？**
- 如何用C 实现 C++ 的面向对象特性（封装、继承、多态）
- **[memcpy](https://www.zhihu.com/search?q=memcpy&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A1868370927})怎么实现让它效率更高？**
- typedef和[define](https://www.zhihu.com/search?q=define&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A1868370927})有什么区别？
- **extern有什么作用，extern C有什么作用？**

**下面是C++基础知识面试题**

- C语言和C++有什么区别？
- **struct和class有什么区别？**
- extern "C"的作用？
- **了解[RAII](https://www.zhihu.com/search?q=RAII&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A2590210440})****吗？介绍一下？RAII可是C++很重要的一个特性。**函数重载和覆盖有什么区别？**谈一谈你对多态的理解，运行时多态的实现原理是什么？**对虚函数机制的理解，[单继承](https://www.zhihu.com/search?q=单继承&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A1868370927})、多继承、[虚继承](https://www.zhihu.com/search?q=虚继承&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A2590210440})条件下虚函数表的结构**如果虚函数是有效的，那为什么不把所有函数设为虚函数？**构造函数可以是虚函数吗？[析构函数](https://www.zhihu.com/search?q=析构函数&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A1868370927})可以是虚函数吗？**基类的析构函数可以调用虚函数吗？基类的构造函数可以调用虚函数吗？**什么场景需要用到[纯虚函数](https://www.zhihu.com/search?q=纯虚函数&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A1868370927})？纯虚函数的作用是什么？**指针和引用有什么区别？什么情况下用指针，什么情况下用引用？**new和[malloc](https://www.zhihu.com/search?q=malloc&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A2590210440})

- 有什么区别？
- **malloc的内存可以用delete释放吗？**
- malloc出来20字节内存，为什么free不需要传入20呢，不会产生内存泄漏吗？
- **new[]和delete[]一定要配对使用吗？为什么？**
- 类的大小怎么计算？
- **volatile关键字的作用**
- 如何实现一个线程池？说一下基本思路即可！
- **了解各种强制类型转换的原理及使用吗？说说？**

C++11新特性基本上在面试中一定会被问到，其实现在[C++14](https://www.zhihu.com/search?q=C%2B%2B14&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A2590210440})，C++17也有很多人在使用。
C++11新特性你都了解多少？可以介绍一下吗？

- 了解auto和[decltype](https://www.zhihu.com/search?q=decltype&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A1868370927})吗？
- **谈一谈你对左值和右值的了解，了解左值引用和[右值引用](https://www.zhihu.com/search?q=右值引用&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A2590210440})****吗？**了解移动语义和完美转发吗？**[enum](https://www.zhihu.com/search?q=enum&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A1868370927}) 和 enum class有什么区别？**了解列表初始化吗？**对[C++11](https://www.zhihu.com/search?q=C%2B%2B11&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A2590210440})****的智能指针了解多少，可以自己实现一个智能指针吗？**平时会用到function、bind、lambda吗，都什么场景下会用到？对C++11的mutex和[RAII lock](https://www.zhihu.com/search?q=RAII lock&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A2590210440})有过了解吗？**一般什么情况下会出现内存泄漏？出现内存泄漏如何调试？**[unique_ptr](https://www.zhihu.com/search?q=unique_ptr&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A2590210440})

- 如何转换的所有权？
- **谈一谈你对面向对象的理解**
- 什么场景下使用继承方式，什么场景下使用组合？

**STL系列**

- C++直接使用数组好还是使用std::array好？std::array是怎么实现的？
- **std::[vector](https://www.zhihu.com/search?q=vector&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A2590210440})****最大的特点是什么？它的内部是怎么实现的？[resize](https://www.zhihu.com/search?q=resize&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A1868370927})和reserve的区别是什么？clear是怎么实现的？**deque的底层[数据结构](https://www.zhihu.com/search?q=数据结构&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A1868370927})是什么？它的内部是怎么实现的？**map和[unordered_map](https://www.zhihu.com/search?q=unordered_map&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A2590210440})**

- **有什么区别？分别在什么场景下使用？**
- list的使用场景？std::find可以传入list对应的迭代器吗？
- **string的常用函数****
- 设计模式，不强求一一列出那23种设计模式，说出几个常见的即可。**

- 分别写出饿汉和懒汉线程安全的[单例模式](https://www.zhihu.com/search?q=单例模式&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A2590210440})
- 说出观察者模式类关系和优点
- **说出代理模式类关系和优点**
- 说出工厂模式概念和优点
- **说出构造者模式概念**
- 说出[适配器模式](https://www.zhihu.com/search?q=适配器模式&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A2590210440})概念

**操作系统**

- 进程和线程的区别？
- **操作系统是怎么进行进程管理的？**
- 操作系统是如何做到进程阻塞的？
- **进程之间的通信方式有哪些？**
- 线程是如何实现的？
- **线程之间私有和共享的资源有哪些？**
- 一般应用程序内存空间的堆和栈的区别是什么？
- **进程虚拟空间是怎么布局的？**
- 虚拟内存是如何映射到物理内存的？了解分页内存管理吗？
- **什么是上下文切换，操作系统是怎么做的上下文切换？**
- 什么是大端字节，什么是小端字节？如何转换字节序？
- **产生死锁的必要条件有哪些？如何避免死锁？**
- 信号和[信号量](https://www.zhihu.com/search?q=信号量&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A1868370927})的区别是什么？
- **锁的性能开销，锁的实现原理？**

**[编译原理](https://www.zhihu.com/search?q=编译原理&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A2590210440})****，编译和链接的知识还是很重要的，解决编译和链接过程中的报错也是C++程序员的基本能力。**

- [gcc](https://www.zhihu.com/search?q=gcc&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A2590210440})hello.c 这行命令具体的执行过程，内部究竟做了什么？
- **程序一定会从main函数开始运行吗？**
- 如何确定某个函数有被编译输出？
- **动态链接库和[静态链接库](https://www.zhihu.com/search?q=静态链接库&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A1868370927})的区别是什么？**