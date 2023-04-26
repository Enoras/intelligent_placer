# intelligent-placer: Собранные данные

## Репрезентативная выборка

### Крайне малый многоугольник: о его размере заранее известно только то, что в него не поместится ни один из объектов выборки.

1.1) Крайне малый многоугольник и малый объект: ответ должен быть False
![test 0_0](0_0.jpg "test 0_0")

1.2) Крайне малый многоугольник и большой объект: ответ должен быть False
![test 0_2](0_2.jpg "test 0_2")

1.3) Крайне малый многоугольник и много небольших объектов: ответ должен быть False
![test 0_3](0_3.jpg "test 0_3")


### Малый многоугольник: заранее известно, что в него из выборки помещается только монета и медиатор.

2.1) Малый многоугольник и малый объект: ответ должен быть True
![test 1_0](1_0.jpg "test 1_0")

2.2) Малый многоугольник и объект, сопоставимых габаритов, но слегка не вписываемый из-за круглой формы: ответ должен быть False
![test 1_1](1_1.jpg "test 1_1")

2.3) Малый многоугольник и большой объект: ответ должен быть False
![test 1_2](1_2.jpg "test 1_2")

### "Хитрый" многоугольник: тупоугольный, но очень длинный треугольник, в который по площади могло бы поместиться много предметов из выборки, но заранее известно, что поместится только жвачка, медиатор и монета (и то, только по одному).

3.1) "Хитрый" многоугольник и малый объект: ответ должен быть True
![test 3_0](3_0.jpg "test 3_0")

3.2) "Хитрый" многоугольник и жвачка: ответ должен быть True
![test 3_3](3_3.jpg "test 3_3")

3.3) "Хитрый" многоугольник и "повернутая" жвачка: ответ должен быть True (т.е. алгоритм должен еще уметь "поворачивать" предметы, чтобы уместить их)
![test 3_4](3_4.jpg "test 3_4")

### Обычный многоугольник: равнобедренный остроугольный треугольник, в который, как заранее известно, могут помещаться несколько маленьких или пара средних объектов.

4.1) Обычный многоугольник и много малых объектов: ответ должен быть True
![test 2_1](2_1.jpg "test 2_1")

4.2) Обычный многоугольник и один средний объект: ответ должен быть True
![test 2_2](2_2.jpg "test 2_2")

### Гигантский многоугольник: заранее известно, что в него могут поместиться абсолютно все предметы выборки за 1 раз (без повторов предметов)

5.1) Гигантский многоугольник и все объекты: ответ должен быть True
![test 4_0](4_0.jpg "test 4_0")

## Остальные тестовые изображения

0.1) Крайне малый многоугольник и пара малых объект: ответ должен быть False
![test 0_1](0_1.jpg "test 0_1")

0.2) Малый многоугольник и много разных объектов: ответ должен быть False
![test 1_3](1_3.jpg "test 1_3")

0.3) "Хитрый" многоугольник и много объектов: ответ должен быть False
![test 3_1](3_1.jpg "test 3_1")

0.4) "Хитрый" многоугольник и небольшой объект, чуть больше монеты: ответ должен быть False
![test 3_2](3_2.jpg "test 3_2")

0.5) Обычный многоугольник и малый объект: ответ должен быть True
![test 2_0](2_0.jpg "test 2_0")

0.6) Обычный многоугольник и много разных объектов: ответ должен быть Falses
![test 2_3](2_3.jpg "test 2_3")

0.7) Гигантский многоугольник и малый объект: ответ должен быть True
![test 4_1](4_1.jpg "test 4_1")

0.8) Гигантский многоугольник и много малых объектов: ответ должен быть True
![test 4_2](4_2.jpg "test 4_2")