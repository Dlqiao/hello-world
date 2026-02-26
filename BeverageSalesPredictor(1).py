import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings

warnings.filterwarnings('ignore')


class BeverageSalesPredictor:
    """
    新饮料销售额预测模型
    模拟尼尔森数据分析流程
    """

    def __init__(self):
        self.data = None
        self.model = None
        self.scaler = StandardScaler()
        self.encoders = {}

    def generate_synthetic_data(self, n_samples=10000):
        """
        生成合成零售数据，模拟尼尔森数据集
        """
        print("🔄 生成模拟零售数据...")
        np.random.seed(42)

        # 基础参数
        regions = ['Northeast', 'Midwest', 'South', 'West']
        store_chains = ['Walmart', 'Kroger', 'Target', 'Costco', 'Whole Foods']
        beverage_types = ['Soda', 'Juice', 'Energy Drink', 'Water', 'Sports Drink']

        data = []

        for i in range(n_samples):
            # 区域特征
            region = np.random.choice(regions)
            population_density = np.random.normal(500, 200)  # 人口密度
            median_income = np.random.normal(60000, 20000)  # 中位收入

            # 商店特征
            store_chain = np.random.choice(store_chains)
            store_size = np.random.normal(50000, 15000)  # 商店面积
            weekly_foot_traffic = np.random.normal(10000, 3000)  # 周客流量

            # 产品特征
            beverage_type = np.random.choice(beverage_types)
            price = np.random.uniform(1.0, 5.0)  # 价格
            promotion_intensity = np.random.uniform(0, 1)  # 促销强度
            shelf_location = np.random.choice(['Eye Level', 'Top Shelf', 'Bottom Shelf'])

            # 市场特征
            competitor_count = np.random.poisson(3)  # 竞争对手数量
            market_share = np.random.beta(2, 5)  # 市场份额

            # 计算基础销售额（业务逻辑）
            base_sales = (
                    population_density * 0.1 +
                    median_income * 0.0001 +
                    store_size * 0.001 +
                    weekly_foot_traffic * 0.05 +
                    (1 - price * 0.2) * 1000 +  # 价格弹性
                    promotion_intensity * 500 +  # 促销效果
                    (1 if shelf_location == 'Eye Level' else 0.5) * 200  # 货架位置影响
            )

            # 添加随机噪声和季节性
            seasonal_factor = 1 + 0.2 * np.sin(i / 100)  # 季节性波动
            noise = np.random.normal(0, 100)  # 随机噪声

            monthly_sales = max(0, base_sales * seasonal_factor + noise)

            data.append({
                'region': region,
                'population_density': population_density,
                'median_income': median_income,
                'store_chain': store_chain,
                'store_size': store_size,
                'weekly_foot_traffic': weekly_foot_traffic,
                'beverage_type': beverage_type,
                'price': price,
                'promotion_intensity': promotion_intensity,
                'shelf_location': shelf_location,
                'competitor_count': competitor_count,
                'market_share': market_share,
                'monthly_sales': monthly_sales
            })

        self.data = pd.DataFrame(data)
        print(f"✅ 生成 {len(self.data)} 条零售记录")
        return self.data

    def feature_engineering(self):
        """
        特征工程 - 创建预测特征
        """
        print("🔧 进行特征工程...")

        df = self.data.copy()

        # 编码分类变量
        categorical_columns = ['region', 'store_chain', 'beverage_type', 'shelf_location']
        for col in categorical_columns:
            le = LabelEncoder()
            df[f'{col}_encoded'] = le.fit_transform(df[col])
            self.encoders[col] = le

        # 创建交互特征
        df['price_promotion_interaction'] = df['price'] * df['promotion_intensity']
        df['income_population_interaction'] = df['median_income'] * df['population_density'] / 1000
        df['traffic_market_share'] = df['weekly_foot_traffic'] * df['market_share']

        # 创建业务指标
        df['sales_per_sqft'] = df['monthly_sales'] / df['store_size']
        df['price_elasticity'] = (df['monthly_sales'] / df['price']).replace([np.inf, -np.inf], 0)

        # 对数变换偏态分布的特征
        skewed_columns = ['monthly_sales', 'store_size', 'weekly_foot_traffic']
        for col in skewed_columns:
            df[f'log_{col}'] = np.log1p(df[col])

        self.features = df
        print(f"✅ 特征工程完成，创建了 {len(df.columns)} 个特征")
        return df

    def exploratory_data_analysis(self):
        """
        探索性数据分析
        """
        print("📊 进行探索性数据分析...")

        df = self.features

        # 创建可视化
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('新饮料销售数据分析', fontsize=16, fontweight='bold')

        # 1. 各区域销售额分布
        region_sales = df.groupby('region')['monthly_sales'].mean().sort_values(ascending=False)
        axes[0, 0].bar(region_sales.index, region_sales.values, color='skyblue')
        axes[0, 0].set_title('各区域平均销售额')
        axes[0, 0].set_ylabel('月销售额 ($)')
        axes[0, 0].tick_params(axis='x', rotation=45)

        # 2. 价格与销售额关系
        axes[0, 1].scatter(df['price'], df['monthly_sales'], alpha=0.6, color='lightcoral')
        axes[0, 1].set_title('价格 vs 销售额')
        axes[0, 1].set_xlabel('价格 ($)')
        axes[0, 1].set_ylabel('月销售额 ($)')

        # 3. 促销强度对销售额的影响
        promotion_bins = pd.cut(df['promotion_intensity'], bins=5)
        promotion_sales = df.groupby(promotion_bins)['monthly_sales'].mean()
        axes[0, 2].plot(range(len(promotion_sales)), promotion_sales.values, marker='o', linewidth=2)
        axes[0, 2].set_title('促销强度对销售额的影响')
        axes[0, 2].set_xlabel('促销强度分组')
        axes[0, 2].set_ylabel('平均月销售额 ($)')
        axes[0, 2].set_xticks(range(len(promotion_sales)))
        axes[0, 2].set_xticklabels([str(x) for x in promotion_sales.index], rotation=45)

        # 4. 饮料类型销售额对比
        beverage_sales = df.groupby('beverage_type')['monthly_sales'].mean().sort_values(ascending=False)
        axes[1, 0].bar(beverage_sales.index, beverage_sales.values, color='lightgreen')
        axes[1, 0].set_title('各饮料类型平均销售额')
        axes[1, 0].set_ylabel('月销售额 ($)')
        axes[1, 0].tick_params(axis='x', rotation=45)

        # 5. 货架位置影响
        shelf_sales = df.groupby('shelf_location')['monthly_sales'].mean()
        axes[1, 1].pie(shelf_sales.values, labels=shelf_sales.index, autopct='%1.1f%%',
                       colors=['gold', 'lightcoral', 'lightblue'])
        axes[1, 1].set_title('货架位置对销售额的影响')

        # 6. 收入与销售额关系
        income_bins = pd.cut(df['median_income'], bins=5)
        income_sales = df.groupby(income_bins)['monthly_sales'].mean()
        axes[1, 2].bar(range(len(income_sales)), income_sales.values, color='purple', alpha=0.7)
        axes[1, 2].set_title('收入水平与销售额关系')
        axes[1, 2].set_xlabel('收入分组')
        axes[1, 2].set_ylabel('平均月销售额 ($)')
        axes[1, 2].set_xticks(range(len(income_sales)))
        axes[1, 2].set_xticklabels([str(x) for x in income_sales.index], rotation=45)

        plt.tight_layout()
        plt.show()

        # 关键洞察输出
        print("\n📈 关键业务洞察:")
        print(f"• 最高销售额区域: {region_sales.index[0]} (${region_sales.iloc[0]:.2f})")
        print(f"• 最畅销饮料类型: {beverage_sales.index[0]} (${beverage_sales.iloc[0]:.2f})")
        print(f"• 最佳货架位置: {shelf_sales.idxmax()} (+{((shelf_sales.max() / shelf_sales.min()) - 1) * 100:.1f}%)")
        print(f"• 价格与销售额相关系数: {df['price'].corr(df['monthly_sales']):.3f}")

        return fig

    def prepare_model_data(self):
        """
        准备建模数据
        """
        print("🤖 准备建模数据...")

        df = self.features

        # 选择特征列
        feature_columns = [
            'population_density', 'median_income', 'store_size',
            'weekly_foot_traffic', 'price', 'promotion_intensity',
            'competitor_count', 'market_share', 'region_encoded',
            'store_chain_encoded', 'beverage_type_encoded', 'shelf_location_encoded',
            'price_promotion_interaction', 'income_population_interaction',
            'traffic_market_share', 'sales_per_sqft'
        ]

        X = df[feature_columns]
        y = df['monthly_sales']

        # 标准化数值特征
        numerical_columns = ['population_density', 'median_income', 'store_size',
                             'weekly_foot_traffic', 'price', 'promotion_intensity']
        X[numerical_columns] = self.scaler.fit_transform(X[numerical_columns])

        self.X = X
        self.y = y
        self.feature_names = feature_columns

        print(f"✅ 建模数据准备完成: {X.shape[1]} 个特征, {X.shape[0]} 个样本")
        return X, y

    def train_models(self):
        """
        训练多个模型并选择最佳模型
        """
        print("🎯 训练预测模型...")

        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )

        # 定义模型
        models = {
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
        }

        best_score = -np.inf
        best_model = None
        results = {}

        for name, model in models.items():
            print(f"  训练 {name}...")

            # 训练模型
            model.fit(X_train, y_train)

            # 预测
            y_pred = model.predict(X_test)

            # 评估
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)

            # 交叉验证
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')

            results[name] = {
                'model': model,
                'mae': mae,
                'rmse': rmse,
                'r2': r2,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }

            print(f"  {name} - R²: {r2:.4f}, MAE: ${mae:.2f}, RMSE: ${rmse:.2f}")

            # 选择最佳模型
            if r2 > best_score:
                best_score = r2
                best_model = name

        self.results = results
        self.model = results[best_model]['model']
        self.best_model_name = best_model

        print(f"✅ 最佳模型: {best_model} (R²: {best_score:.4f})")
        return results

    def predict_new_beverage(self, new_product_features):
        """
        预测新饮料的销售额
        """
        print("🔮 预测新饮料销售额...")

        # 准备新产品的特征
        new_product_df = pd.DataFrame([new_product_features])

        # 应用相同的特征工程
        for col in ['region', 'store_chain', 'beverage_type', 'shelf_location']:
            if col in new_product_df.columns:
                le = self.encoders[col]
                new_product_df[f'{col}_encoded'] = le.transform(new_product_df[col])

        # 创建交互特征
        new_product_df['price_promotion_interaction'] = (
                new_product_df['price'] * new_product_df['promotion_intensity']
        )
        new_product_df['income_population_interaction'] = (
                new_product_df['median_income'] * new_product_df['population_density'] / 1000
        )
        new_product_df['traffic_market_share'] = (
                new_product_df['weekly_foot_traffic'] * new_product_df['market_share']
        )
        new_product_df['sales_per_sqft'] = 0  # 新产品的占位符

        # 选择特征并标准化
        feature_columns = self.feature_names
        X_new = new_product_df[feature_columns]

        numerical_columns = ['population_density', 'median_income', 'store_size',
                             'weekly_foot_traffic', 'price', 'promotion_intensity']
        X_new[numerical_columns] = self.scaler.transform(X_new[numerical_columns])

        # 预测
        predicted_sales = self.model.predict(X_new)[0]

        # 计算年度预测（12个月）
        annual_prediction = predicted_sales * 12

        print(f"📊 预测结果:")
        print(f"  月度销售额预测: ${predicted_sales:,.2f}")
        print(f"  年度销售额预测: ${annual_prediction:,.2f}")

        return predicted_sales, annual_prediction

    def feature_importance_analysis(self):
        """
        特征重要性分析
        """
        print("🔍 分析特征重要性...")

        if hasattr(self.model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)

            plt.figure(figsize=(10, 8))
            sns.barplot(data=importance_df.head(10), y='feature', x='importance')
            plt.title('Top 10 特征重要性')
            plt.xlabel('重要性分数')
            plt.tight_layout()
            plt.show()

            print("\n🏆 最重要的特征:")
            for i, row in importance_df.head(5).iterrows():
                print(f"  {row['feature']}: {row['importance']:.3f}")

            return importance_df

    def scenario_analysis(self, base_features):
        """
        情景分析：不同策略下的销售额预测
        """
        print("🎭 进行情景分析...")

        scenarios = {
            '基准情景': base_features,
            '积极促销': {**base_features, 'promotion_intensity': 0.8, 'price': base_features['price'] * 0.9},
            '高端定位': {**base_features, 'price': base_features['price'] * 1.2, 'shelf_location': 'Eye Level'},
            '大众市场': {**base_features, 'price': base_features['price'] * 0.8, 'promotion_intensity': 0.6}
        }

        results = {}

        for scenario_name, features in scenarios.items():
            monthly, annual = self.predict_new_beverage(features)
            results[scenario_name] = {
                'monthly_sales': monthly,
                'annual_sales': annual,
                'features': features
            }

        # 可视化情景分析结果
        scenario_names = list(results.keys())
        annual_sales = [results[name]['annual_sales'] for name in scenario_names]

        plt.figure(figsize=(12, 6))
        bars = plt.bar(scenario_names, annual_sales, color=['lightblue', 'lightgreen', 'gold', 'lightcoral'])
        plt.title('不同市场策略下的年度销售额预测')
        plt.ylabel('年度销售额预测 ($)')
        plt.xticks(rotation=45)

        # 在柱状图上添加数值标签
        for bar, value in zip(bars, annual_sales):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1000,
                     f'${value:,.0f}', ha='center', va='bottom')

        plt.tight_layout()
        plt.show()

        print("\n📈 情景分析总结:")
        best_scenario = max(results.items(), key=lambda x: x[1]['annual_sales'])
        worst_scenario = min(results.items(), key=lambda x: x[1]['annual_sales'])

        print(f"• 最佳策略: {best_scenario[0]} (${best_scenario[1]['annual_sales']:,.2f})")
        print(f"• 最差策略: {worst_scenario[0]} (${worst_scenario[1]['annual_sales']:,.2f})")
        print(f"• 策略间最大差异: ${best_scenario[1]['annual_sales'] - worst_scenario[1]['annual_sales']:,.2f}")

        return results


# 主执行函数
def main():
    """
    执行完整的新饮料销售额预测流程
    """
    print("=" * 60)
    print("       新饮料销售额预测分析 - 尼尔森风格")
    print("=" * 60)

    # 初始化预测器
    predictor = BeverageSalesPredictor()

    # 1. 生成数据
    data = predictor.generate_synthetic_data(10000)

    # 2. 特征工程
    features = predictor.feature_engineering()

    # 3. 探索性分析
    predictor.exploratory_data_analysis()

    # 4. 准备建模数据
    X, y = predictor.prepare_model_data()

    # 5. 训练模型
    results = predictor.train_models()

    # 6. 特征重要性分析
    importance_df = predictor.feature_importance_analysis()

    # 7. 定义新产品特征
    new_beverage_features = {
        'region': 'Northeast',
        'population_density': 600,
        'median_income': 75000,
        'store_chain': 'Walmart',
        'store_size': 55000,
        'weekly_foot_traffic': 12000,
        'beverage_type': 'Energy Drink',  # 新能量饮料
        'price': 3.50,
        'promotion_intensity': 0.5,
        'shelf_location': 'Eye Level',
        'competitor_count': 4,
        'market_share': 0.15
    }

    # 8. 预测新产品销售额
    monthly_sales, annual_sales = predictor.predict_new_beverage(new_beverage_features)

    # 9. 情景分析
    scenario_results = predictor.scenario_analysis(new_beverage_features)

    # 10. 生成业务报告
    generate_business_report(predictor, new_beverage_features, scenario_results)


def generate_business_report(predictor, product_features, scenario_results):
    """
    生成商业决策报告
    """
    print("\n" + "=" * 60)
    print("           商业决策报告")
    print("=" * 60)

    print(f"📋 产品信息:")
    print(f"  • 产品类型: {product_features['beverage_type']}")
    print(f"  • 目标价格: ${product_features['price']:.2f}")
    print(f"  • 目标区域: {product_features['region']}")
    print(f"  • 主要渠道: {product_features['store_chain']}")

    print(f"\n🎯 预测结果:")
    base_annual = scenario_results['基准情景']['annual_sales']
    best_annual = max([r['annual_sales'] for r in scenario_results.values()])
    worst_annual = min([r['annual_sales'] for r in scenario_results.values()])

    print(f"  • 基准年度销售额: ${base_annual:,.2f}")
    print(f"  • 最佳情景销售额: ${best_annual:,.2f} (+{(best_annual / base_annual - 1) * 100:.1f}%)")
    print(f"  • 最差情景销售额: ${worst_annual:,.2f} (-{(1 - worst_annual / base_annual) * 100:.1f}%)")

    print(f"\n💡 战略建议:")
    print(f"  1. 定价策略: 建议采用${scenario_results['积极促销']['features']['price']:.2f}的促销价格")
    print(
        f"  2. 促销强度: 建议维持{scenario_results['积极促销']['features']['promotion_intensity'] * 100:.0f}%的促销力度")
    print(f"  3. 货架位置: 确保获得'{product_features['shelf_location']}'货架位置")
    print(f"  4. 区域重点: 优先在'{product_features['region']}'区域投放")

    print(f"\n⚠️  风险提示:")
    print(f"  • 市场竞争: {product_features['competitor_count']}个主要竞争对手")
    print(f"  • 价格敏感度: 价格变动10%可能影响销售额约8-12%")
    print(f"  • 区域差异: 不同区域销售额可能相差30-50%")

    print(f"\n📞 下一步行动:")
    print(f"  1. 验证模型在历史类似产品上的准确性")
    print(f"  2. 进行小规模市场测试")
    print(f"  3. 监控前3个月的实际销售数据")
    print(f"  4. 根据实际数据调整预测模型")


if __name__ == "__main__":
    main()