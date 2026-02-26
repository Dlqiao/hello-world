#!/usr/bin/env python3
"""
全球零售数据分析 - 修复相关系数错误
"""
import spark_env
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql.window import Window
import os
import tempfile
import pandas as pd
from datetime import datetime


class GlobalSuperstoreAnalysisComplete:
    def __init__(self, output_base_path=None):
        self.spark = self.create_spark_session()
        self.dataset_path = "/Users/dongliqiao/.cache/kagglehub/datasets/anandaramg/global-superstore/versions/1"

        # 设置输出路径
        if output_base_path is None:
            self.output_base_path = f"/Users/dongliqiao/Desktop/global_superstore_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        else:
            self.output_base_path = output_base_path

        # 创建输出目录
        os.makedirs(self.output_base_path, exist_ok=True)
        print(f"输出目录: {self.output_base_path}")

    def create_spark_session(self):
        """创建 SparkSession"""
        return SparkSession.builder \
            .appName("GlobalSuperstoreAnalysis") \
            .config("spark.sql.adaptive.enabled", "true") \
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
            .getOrCreate()

    def find_csv_file(self):
        """在指定路径中查找 CSV 文件"""
        print(f"在路径中查找 CSV 文件: {self.dataset_path}")

        if not os.path.exists(self.dataset_path):
            print(f"错误: 路径不存在: {self.dataset_path}")
            return None

        # 查找 CSV 文件
        csv_files = []
        for file in os.listdir(self.dataset_path):
            if file.lower().endswith('.csv'):
                csv_files.append(os.path.join(self.dataset_path, file))
                print(f"找到 CSV 文件: {file}")

        if not csv_files:
            print("未找到 CSV 文件")
            return None

        selected_file = csv_files[0]
        print(f"选择文件: {selected_file}")
        return selected_file

    def load_global_superstore_data(self):
        """加载 Global Superstore 数据"""
        csv_file = self.find_csv_file()
        if csv_file is None:
            return None

        try:
            # 读取 CSV 文件
            df = self.spark.read \
                .option("header", "true") \
                .option("inferSchema", "true") \
                .csv(csv_file)

            print(f"数据加载成功: {df.count()} 行, {len(df.columns)} 列")
            print(f"数据列: {df.columns}")

            # 显示数据结构
            print("\n数据结构:")
            df.printSchema()

            print("\n数据示例 (前5行):")
            df.show(5, truncate=False)

            return df

        except Exception as e:
            print(f"加载数据失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def clean_global_superstore_data(self, df):
        """清洗 Global Superstore 数据"""
        print("\n=== 数据清洗和预处理 ===")

        initial_count = df.count()
        print(f"初始记录数: {initial_count:,}")

        # 数据质量检查和处理 - 使用正确的列名
        df_clean = df.filter(
            (col("Sales") > 0) &
            (col("Quantity") > 0) &
            (col("Profit").isNotNull())
        )

        # 数据类型转换和衍生列 - 使用正确的列名
        df_clean = df_clean \
            .withColumn("Order Date", to_date(col("Order Date"), "MM/dd/yyyy")) \
            .withColumn("Ship Date", to_date(col("Ship Date"), "MM/dd/yyyy")) \
            .withColumn("Profit_Margin", round((col("Profit") / col("Sales")) * 100, 2)) \
            .withColumn("Order_Year", year(col("Order Date"))) \
            .withColumn("Order_Month", month(col("Order Date"))) \
            .withColumn("Order_Quarter", quarter(col("Order Date"))) \
            .withColumn("YearMonth", date_format(col("Order Date"), "yyyy-MM")) \
            .withColumn("Shipping_Days", datediff(col("Ship Date"), col("Order Date")))

        # 处理可能的空值 - 只处理实际存在的列
        fill_values = {"Discount": 0.0}
        if "Shipping Cost" in df_clean.columns:
            fill_values["Shipping Cost"] = 0.0

        df_clean = df_clean.fillna(fill_values)

        cleaned_count = df_clean.count()
        print(f"清洗后记录数: {cleaned_count:,}")
        print(f"数据保留率: {(cleaned_count / initial_count) * 100:.2f}%")

        # 显示清洗后的数据示例
        print("\n清洗后数据示例:")
        df_clean.select("Order ID", "Order Date", "Sales", "Profit", "Profit_Margin").show(5)

        return df_clean

    def data_quality_report(self, df):
        """数据质量报告 - 修复 isnan 错误"""
        print("\n" + "=" * 60)
        print("数据质量报告")
        print("=" * 60)

        # 基本统计
        print("\n1. 数据集概览:")
        summary = df.agg(
            count("*").alias("总记录数"),
            countDistinct("Order ID").alias("唯一订单数"),
            countDistinct("Customer ID").alias("唯一客户数"),
            countDistinct("Product ID").alias("唯一产品数"),
            countDistinct("Country").alias("国家数量")
        )
        summary.show()

        # 空值检查 - 修复 isnan 错误
        print("\n2. 空值检查 (前10列):")
        # 只检查前10列，避免输出太长
        columns_to_check = df.columns[:10]

        # 创建空值检查表达式 - 只对数值列使用 isnan
        null_expressions = []
        for column in columns_to_check:
            # 获取列的数据类型
            col_type = dict(df.dtypes)[column]

            # 如果是数值类型，检查 isnan 和 isNull
            if col_type in ['double', 'float', 'int', 'bigint', 'decimal']:
                null_expressions.append(
                    count(when(col(column).isNull() | isnan(col(column)), column)).alias(column)
                )
            else:
                # 对于非数值类型，只检查 isNull
                null_expressions.append(
                    count(when(col(column).isNull(), column)).alias(column)
                )

        null_check = df.select(null_expressions)
        null_check.show(vertical=True)

        # 数值列统计
        print("\n3. 数值列统计:")
        numeric_columns = ["Sales", "Quantity", "Discount", "Profit", "Shipping Cost"]
        # 只检查存在的数值列
        existing_numeric_cols = [col for col in numeric_columns if col in df.columns]
        if existing_numeric_cols:
            df.describe(*existing_numeric_cols).show()
        else:
            print("没有找到数值列")

        return summary

    def regional_analysis(self, df):
        """区域销售分析"""
        print("\n" + "=" * 60)
        print("区域销售分析")
        print("=" * 60)

        # 区域表现分析
        print("\n1. 区域表现排名:")
        region_stats = df.groupBy("Region") \
            .agg(
            round(sum("Sales"), 2).alias("总销售额"),
            round(sum("Profit"), 2).alias("总利润"),
            countDistinct("Customer ID").alias("客户数"),
            countDistinct("Order ID").alias("订单数"),
            round(avg("Sales"), 2).alias("平均订单金额"),
            round((sum("Profit") / sum("Sales")) * 100, 2).alias("利润率")
        ) \
            .orderBy(desc("总销售额"))
        region_stats.show()

        # 市场分析
        if "Market" in df.columns:
            print("\n2. 市场表现排名:")
            market_stats = df.groupBy("Market") \
                .agg(
                round(sum("Sales"), 2).alias("总销售额"),
                round(sum("Profit"), 2).alias("总利润"),
                countDistinct("Customer ID").alias("客户数")
            ) \
                .orderBy(desc("总销售额"))
            market_stats.show()
        else:
            market_stats = None
            print("\n2. 市场分析: 数据集中没有 Market 列")

        # 国家分析
        print("\n3. 国家销售额TOP 10:")
        country_stats = df.groupBy("Country") \
            .agg(
            round(sum("Sales"), 2).alias("总销售额"),
            round(sum("Profit"), 2).alias("总利润"),
            countDistinct("Customer ID").alias("客户数")
        ) \
            .orderBy(desc("总销售额")) \
            .limit(10)
        country_stats.show()

        return region_stats, market_stats, country_stats

    def product_analysis(self, df):
        """产品类别分析"""
        print("\n" + "=" * 60)
        print("产品类别分析")
        print("=" * 60)

        # 产品类别分析
        print("\n1. 产品类别表现:")
        category_stats = df.groupBy("Category", "Sub-Category") \
            .agg(
            round(sum("Sales"), 2).alias("总销售额"),
            round(sum("Profit"), 2).alias("总利润"),
            count("*").alias("销售数量"),
            round(avg("Sales"), 2).alias("平均售价"),
            round((sum("Profit") / sum("Sales")) * 100, 2).alias("利润率"),
            round(avg("Discount") * 100, 2).alias("平均折扣率")
        ) \
            .orderBy(desc("总销售额"))
        category_stats.show(truncate=False)

        # 最畅销产品
        print("\n2. 最畅销产品TOP 10:")
        top_products = df.groupBy("Product ID", "Product Name", "Category") \
            .agg(
            round(sum("Sales"), 2).alias("总销售额"),
            sum("Quantity").alias("总销量"),
            countDistinct("Order ID").alias("购买次数"),
            round(avg("Sales"), 2).alias("平均售价")
        ) \
            .orderBy(desc("总销售额")) \
            .limit(10)
        top_products.show(truncate=False)

        # 最盈利产品
        print("\n3. 最盈利产品TOP 10:")
        top_profitable = df.groupBy("Product ID", "Product Name", "Category") \
            .agg(
            round(sum("Profit"), 2).alias("总利润"),
            round(sum("Sales"), 2).alias("总销售额"),
            round((sum("Profit") / sum("Sales")) * 100, 2).alias("利润率")
        ) \
            .filter(col("总利润") > 0) \
            .orderBy(desc("总利润")) \
            .limit(10)
        top_profitable.show(truncate=False)

        return category_stats, top_products, top_profitable

    def time_series_analysis(self, df):
        """时间序列分析"""
        print("\n" + "=" * 60)
        print("时间序列分析")
        print("=" * 60)

        # 年度趋势
        print("\n1. 年度销售趋势:")
        yearly_sales = df.groupBy("Order_Year") \
            .agg(
            round(sum("Sales"), 2).alias("年销售额"),
            round(sum("Profit"), 2).alias("年利润"),
            countDistinct("Customer ID").alias("年活跃客户"),
            countDistinct("Order ID").alias("年订单数"),
            round((sum("Profit") / sum("Sales")) * 100, 2).alias("年利润率")
        ) \
            .orderBy("Order_Year")
        yearly_sales.show()

        # 月度趋势
        print("\n2. 月度销售趋势:")
        monthly_sales = df.groupBy("YearMonth") \
            .agg(
            round(sum("Sales"), 2).alias("月销售额"),
            round(sum("Profit"), 2).alias("月利润"),
            countDistinct("Order ID").alias("月订单数")
        ) \
            .orderBy("YearMonth")
        monthly_sales.show(12, truncate=False)

        # 季度分析
        print("\n3. 季度分析:")
        quarterly_sales = df.groupBy("Order_Year", "Order_Quarter") \
            .agg(
            round(sum("Sales"), 2).alias("季度销售额"),
            round(sum("Profit"), 2).alias("季度利润")
        ) \
            .orderBy("Order_Year", "Order_Quarter")
        quarterly_sales.show()

        # 周分析 (如果存在weeknum列)
        if "weeknum" in df.columns:
            print("\n4. 周销售分析 (TOP 15):")
            weekly_sales = df.groupBy("Year", "weeknum") \
                .agg(
                round(sum("Sales"), 2).alias("周销售额"),
                round(sum("Profit"), 2).alias("周利润"),
                countDistinct("Order ID").alias("周订单数")
            ) \
                .orderBy(desc("周销售额")) \
                .limit(15)
            weekly_sales.show()
        else:
            weekly_sales = None
            print("\n4. 周分析: 数据集中没有 weeknum 列")

        return yearly_sales, monthly_sales, quarterly_sales, weekly_sales

    def customer_analysis(self, df):
        """客户分析"""
        print("\n" + "=" * 60)
        print("客户分析")
        print("=" * 60)

        # 客户细分分析
        print("\n1. 客户细分表现:")
        segment_stats = df.groupBy("Segment") \
            .agg(
            countDistinct("Customer ID").alias("客户数"),
            round(sum("Sales"), 2).alias("总销售额"),
            round(sum("Profit"), 2).alias("总利润"),
            round(avg("Sales"), 2).alias("客户平均消费"),
            round((sum("Profit") / sum("Sales")) * 100, 2).alias("细分利润率")
        ) \
            .orderBy(desc("总销售额"))
        segment_stats.show()

        # 高价值客户识别
        print("\n2. 高价值客户TOP 10:")
        top_customers = df.groupBy("Customer ID", "Customer Name", "Segment") \
            .agg(
            round(sum("Sales"), 2).alias("总消费金额"),
            countDistinct("Order ID").alias("订单数"),
            round(avg("Sales"), 2).alias("平均订单金额"),
            round(sum("Profit"), 2).alias("总利润"),
            round(avg("Shipping_Days"), 2).alias("平均运输天数")
        ) \
            .orderBy(desc("总消费金额")) \
            .limit(10)
        top_customers.show(truncate=False)

        return segment_stats, top_customers

    def shipping_analysis(self, df):
        """运输模式分析 - 修复相关系数错误"""
        print("\n" + "=" * 60)
        print("运输模式分析")
        print("=" * 60)

        shipping_stats = df.groupBy("Ship Mode") \
            .agg(
            count("Order ID").alias("订单数量"),
            round(sum("Sales"), 2).alias("总销售额"),
            round(avg("Sales"), 2).alias("平均订单金额"),
            round(avg("Profit"), 2).alias("平均利润"),
            round(avg("Shipping_Days"), 2).alias("平均运输天数"),
            round(avg("Profit_Margin"), 2).alias("平均利润率")
        ) \
            .orderBy(desc("订单数量"))

        print("运输模式表现分析:")
        shipping_stats.show()

        # 运输时间与销售额的关系 - 修复相关系数错误
        print("\n运输时间分析:")
        try:
            # 确保两列都是数值类型
            shipping_corr = df.stat.corr("Shipping_Days", "Sales")
            print(f"运输天数与销售额的相关系数: {shipping_corr:.4f}")
        except Exception as e:
            print(f"无法计算相关系数: {e}")
            print("这可能是因为数据中存在非数值类型或空值")

        # 运输成本分析 (如果存在Shipping Cost列)
        if "Shipping Cost" in df.columns:
            print("\n运输成本分析:")
            shipping_cost_stats = df.agg(
                round(avg("Shipping Cost"), 2).alias("平均运输成本"),
                round(max("Shipping Cost"), 2).alias("最高运输成本"),
                round(min("Shipping Cost"), 2).alias("最低运输成本"),
                round(sum("Shipping Cost"), 2).alias("总运输成本")
            )
            shipping_cost_stats.show()
        else:
            shipping_cost_stats = None
            print("\n运输成本分析: 数据集中没有 Shipping Cost 列")

        return shipping_stats

    def save_dataframe(self, df, name, format_type="csv", single_file=True):
        """
        保存 DataFrame 到文件
        """
        save_path = os.path.join(self.output_base_path, name)

        try:
            # 准备写入器
            writer = df.write.mode("overwrite")

            # 如果是 CSV，添加 header 选项
            if format_type == "csv":
                writer = writer.option("header", "true")

            # 如果要求单个文件，先合并分区
            if single_file:
                df = df.coalesce(1)

            # 执行保存
            if format_type == "csv":
                writer.csv(save_path)
            elif format_type == "parquet":
                writer.parquet(save_path)
            elif format_type == "json":
                writer.json(save_path)
            else:
                print(f"不支持的格式: {format_type}")
                return False

            print(f"✅ 已保存: {name}.{format_type}")
            return True

        except Exception as e:
            print(f"❌ 保存失败 {name}: {e}")
            return False

    def save_as_excel(self, df, name):
        """保存为 Excel 文件（适合小数据集）"""
        try:
            # 转换为 pandas DataFrame
            pandas_df = df.toPandas()

            # 保存为 Excel
            excel_path = os.path.join(self.output_base_path, f"{name}.xlsx")
            pandas_df.to_excel(excel_path, index=False, engine='openpyxl')

            print(f"✅ 已保存为 Excel: {name}.xlsx")
            return True

        except Exception as e:
            print(f"❌ Excel 保存失败 {name}: {e}")
            return False

    def save_analysis_results(self, results_dict):
        """保存所有分析结果"""
        print("\n" + "=" * 60)
        print("开始保存分析结果")
        print("=" * 60)

        # 保存主要数据集
        if 'cleaned_data' in results_dict:
            self.save_dataframe(results_dict['cleaned_data'], "cleaned_dataset", "parquet")
            self.save_dataframe(results_dict['cleaned_data'], "cleaned_dataset", "csv")
            # 限制行数保存 Excel 样本
            sample_df = results_dict['cleaned_data'].limit(100000) if results_dict['cleaned_data'].count() > 100000 else \
            results_dict['cleaned_data']
            self.save_as_excel(sample_df, "cleaned_dataset_sample")

        # 保存分析结果
        for name, df in results_dict.items():
            if name != 'cleaned_data' and df is not None:  # 已单独处理且不为空
                self.save_dataframe(df, f"analysis_{name}", "csv")
                # 限制行数保存 Excel 样本
                sample_df = df.limit(50000) if df.count() > 50000 else df
                self.save_as_excel(sample_df, f"analysis_{name}")

        print(f"\n🎉 所有分析结果已保存到: {self.output_base_path}")

    def generate_comprehensive_report(self, df, analysis_results):
        """生成综合分析报告"""
        print("\n" + "=" * 80)
        print("Global Superstore 综合分析报告")
        print("=" * 80)

        # 关键指标汇总
        total_metrics = df.agg(
            count("*").alias("总交易数"),
            countDistinct("Order ID").alias("唯一订单数"),
            countDistinct("Customer ID").alias("唯一客户数"),
            countDistinct("Product ID").alias("唯一产品数"),
            round(sum("Sales"), 2).alias("总销售额"),
            round(sum("Profit"), 2).alias("总利润"),
            round(avg("Sales"), 2).alias("平均订单金额"),
            round((sum("Profit") / sum("Sales")) * 100, 2).alias("整体利润率")
        ).collect()[0]

        print("\n📊 关键业务指标:")
        print(f"   • 总交易数: {total_metrics['总交易数']:,}")
        print(f"   • 唯一客户数: {total_metrics['唯一客户数']:,}")
        print(f"   • 唯一产品数: {total_metrics['唯一产品数']:,}")
        print(f"   • 总销售额: ${total_metrics['总销售额']:,.2f}")
        print(f"   • 总利润: ${total_metrics['总利润']:,.2f}")
        print(f"   • 平均订单金额: ${total_metrics['平均订单金额']:.2f}")
        print(f"   • 整体利润率: {total_metrics['整体利润率']}%")

        # 如果存在运输成本列，添加运输成本信息
        if "Shipping Cost" in df.columns:
            shipping_total = df.agg(round(sum("Shipping Cost"), 2).alias("总运输成本")).collect()[0]["总运输成本"]
            print(f"   • 总运输成本: ${shipping_total:,.2f}")

        # 区域表现
        best_region = df.groupBy("Region") \
            .agg(round(sum("Profit"), 2).alias("Region_Profit")) \
            .orderBy(desc("Region_Profit")) \
            .limit(1) \
            .collect()[0]

        print(f"\n🌍 最佳表现区域: {best_region['Region']} (${best_region['Region_Profit']:,.2f})")

        # 市场表现 (如果存在Market列)
        if "Market" in df.columns:
            best_market = df.groupBy("Market") \
                .agg(round(sum("Profit"), 2).alias("Market_Profit")) \
                .orderBy(desc("Market_Profit")) \
                .limit(1) \
                .collect()[0]

            print(f"🏪 最佳表现市场: {best_market['Market']} (${best_market['Market_Profit']:,.2f})")

        # 产品洞察
        best_category = df.groupBy("Category") \
            .agg(round((sum("Profit") / sum("Sales")) * 100, 2).alias("Category_Margin")) \
            .orderBy(desc("Category_Margin")) \
            .limit(1) \
            .collect()[0]

        print(f"📦 最盈利品类: {best_category['Category']} ({best_category['Category_Margin']}% 利润率)")

        # 客户洞察
        avg_customer_value = df.groupBy("Customer ID") \
            .agg(round(sum("Sales"), 2).alias("Customer_Value")) \
            .agg(round(avg("Customer_Value"), 2).alias("Avg_Customer_Value")) \
            .collect()[0]["Avg_Customer_Value"]

        print(f"👥 客户平均价值: ${avg_customer_value:,.2f}")

        print("\n🎯 业务洞察建议:")
        print("   1. 重点关注高价值客户群体，提高客户留存率")
        print("   2. 优化产品组合，提高整体利润率")
        print("   3. 分析区域表现，优化市场资源分配")
        print("   4. 监控销售趋势，及时调整库存和促销策略")
        print("   5. 优化运输模式，平衡成本和服务水平")

        # 保存报告到文件
        report_path = os.path.join(self.output_base_path, "analysis_report.txt")
        with open(report_path, 'w') as f:
            f.write("Global Superstore 分析报告\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"总交易数: {total_metrics['总交易数']:,}\n")
            f.write(f"总销售额: ${total_metrics['总销售额']:,.2f}\n")
            f.write(f"总利润: ${total_metrics['总利润']:,.2f}\n")
            f.write(f"整体利润率: {total_metrics['整体利润率']}%\n")
            f.write(f"最佳表现区域: {best_region['Region']} (${best_region['Region_Profit']:,.2f})\n")
            if "Market" in df.columns:
                f.write(f"最佳表现市场: {best_market['Market']} (${best_market['Market_Profit']:,.2f})\n")
            f.write(f"最盈利品类: {best_category['Category']} ({best_category['Category_Margin']}%)\n")
            f.write(f"客户平均价值: ${avg_customer_value:,.2f}\n")
            f.write(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        print(f"\n📄 详细报告已保存: {report_path}")

    def run_complete_analysis(self):
        """运行完整分析流程"""
        try:
            print("开始 Global Superstore 数据分析...")

            # 1. 加载数据
            print("\n步骤 1/6: 加载数据")
            df_raw = self.load_global_superstore_data()

            if df_raw is None:
                print("数据加载失败，分析终止")
                return None

            # 2. 数据清洗
            print("\n步骤 2/6: 数据清洗和预处理")
            df_clean = self.clean_global_superstore_data(df_raw)

            # 3. 数据质量报告
            print("\n步骤 3/6: 数据质量分析")
            quality_report = self.data_quality_report(df_clean)

            # 4. 业务分析
            print("\n步骤 4/6: 业务分析")
            region_stats, market_stats, country_stats = self.regional_analysis(df_clean)
            category_stats, top_products, top_profitable = self.product_analysis(df_clean)
            yearly_sales, monthly_sales, quarterly_sales, weekly_sales = self.time_series_analysis(df_clean)
            segment_stats, top_customers = self.customer_analysis(df_clean)
            shipping_stats = self.shipping_analysis(df_clean)

            # 5. 保存结果
            print("\n步骤 5/6: 保存分析结果")
            analysis_results = {
                'cleaned_data': df_clean,
                'regional_analysis': region_stats,
                'market_analysis': market_stats,
                'country_analysis': country_stats,
                'product_analysis': category_stats,
                'top_products': top_products,
                'top_profitable_products': top_profitable,
                'yearly_sales': yearly_sales,
                'monthly_sales': monthly_sales,
                'quarterly_sales': quarterly_sales,
                'weekly_sales': weekly_sales,
                'customer_segments': segment_stats,
                'top_customers': top_customers,
                'shipping_analysis': shipping_stats
            }

            self.save_analysis_results(analysis_results)

            # 6. 生成报告
            print("\n步骤 6/6: 生成综合报告")
            self.generate_comprehensive_report(df_clean, analysis_results)

            print(f"\n✅ 分析完成!")

            return analysis_results

        except Exception as e:
            print(f"❌ 分析过程中出现错误: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def list_saved_files(self):
        """列出所有保存的文件"""
        print("\n" + "=" * 60)
        print("保存的文件列表")
        print("=" * 60)

        for root, dirs, files in os.walk(self.output_base_path):
            for file in files:
                file_path = os.path.join(root, file)
                file_size = os.path.getsize(file_path)
                print(f"📄 {file} ({file_size:,} bytes)")


def main():
    """主函数"""
    # 可以指定自定义输出路径，或使用默认路径
    custom_output_path = "/Users/dongliqiao/Desktop/global_superstore_results"

    analyzer = GlobalSuperstoreAnalysisComplete(output_base_path=custom_output_path)

    try:
        results = analyzer.run_complete_analysis()

        if results:
            print("\n🎉 分析完成!")
            analyzer.list_saved_files()

            # 在Finder中打开结果目录
            os.system(f"open {analyzer.output_base_path}")
        else:
            print("\n⚠️ 分析失败")

    except Exception as e:
        print(f"\n❌ 分析过程中出错: {e}")
    finally:
        analyzer.spark.stop()


if __name__ == "__main__":
    main()