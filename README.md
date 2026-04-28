# Toy-ParaRNN MVP: parity task через систему нелинейных уравнений

## Идея проекта

Это учебная демонстрация идеи по мотивам ParaRNN на простой toy-задаче.

Обычная RNN считает скрытые состояния последовательно:

```text
h_0 -> h_1 -> ... -> h_L
```

В этом проекте мы переписываем всю цепочку как систему нелинейных уравнений:

```text
F(h) = 0
```

где неизвестные:

```text
h = [h_1, h_2, ..., h_L]
```

Затем мы решаем эту систему методом Ньютона. На шаге Ньютона возникает линейная рекурсия для поправок, а эту рекурсию можно решить либо обычным последовательным циклом, либо демонстрационным scan/doubling.

Это не реализация настоящей Apple ParaRNN. Это toy-проект для понимания идеи.

## Задача parity task

На вход подаётся бинарная последовательность из `0` и `1`. Нужно определить, чётное или нечётное количество единиц.

Примеры:

```text
0 1 0 1 -> две единицы -> ответ 0
1 0 1 1 -> три единицы -> ответ 1
```

## Toy-RNN

Мы используем преобразование входа:

```text
s_t = 1 - 2x_t
```

Начальное состояние:

```text
h_0 = 1
```

Рекуррентное правило:

```text
h_t = tanh(alpha * s_t * h_{t-1})
```

Интерпретация знака:

```text
h_t > 0 означает чётную parity префикса
h_t < 0 означает нечётную parity префикса
```

## Что реализовано

- sequential RNN;
- residual `F(h)`;
- проверка, что sequential RNN даёт `F(h)≈0`;
- Newton solver;
- линейная рекурсия для Newton step;
- dense Jacobian check;
- sequential linear solve;
- toy scan/doubling solve;
- batch accuracy;
- hidden state plot;
- residual decay plot;
- dependency depth plot;
- runtime benchmark;
- alpha experiment.

## Как запустить

```bash
pip install -r requirements.txt
jupyter notebook toy_pararnn_parity_mvp.ipynb
```

Затем в Jupyter:

```text
Kernel -> Restart Kernel and Run All Cells
```

## Ограничения

- нет обучения большой нейросети;
- нет CUDA;
- нет настоящего GPU parallel scan;
- hidden state одномерный;
- parity task очень простая;
- Python scan/doubling не обязан быть быстрее обычной RNN;
- проект показывает идею, а не воспроизводит Apple ParaRNN;
- parallel scan применяется к линеаризованной задаче на шаге Ньютона, а не напрямую к исходной нелинейной RNN.
