# Data

`example/ingredient_database_example.csv` is a synthetic ingredient database with the columns required by the code. It contains the 25 ingredients of the case study, but prices and life-cycle coefficients were randomly scaled and the compositions slightly perturbed. It is meant for running and testing the code and must not be used as data.

`example/adc_largemouth_bass.csv` lists digestibility coefficients measured in largemouth bass for 9 ingredients, taken from the studies in `example/adc_largemouth_bass_references.csv`. They are used only to check the database values.

To run the case study, save the ingredient database as `data/ingredient_database.csv`.

# 数据

`example/ingredient_database_example.csv` 为合成的原料数据库，列与程序要求一致。原料与案例相同，但价格和生命周期系数经过随机缩放，成分经过小幅扰动，仅用于运行和测试代码，不能作为数据使用。

`example/adc_largemouth_bass.csv` 为 9 种原料在大口黑鲈中实测的消化率，摘自 `example/adc_largemouth_bass_references.csv` 所列文献，仅用于核对数据库数值。

运行论文案例时，将原料数据库保存为 `data/ingredient_database.csv`。
