import mysql.connector

def save_predict_history(result, customer_id="NEW"):
    try:
        conn = mysql.connector.connect(
            host="localhost",
            user="root",
            password="@Obama123",
            database="chulibi"
        )
        cur = conn.cursor()

        sql = """
        INSERT INTO history_predict 
        (customer_id, cluster, proba_churn, pred_churn, churn_risk_pct, expected_loss,
         action_id, action_name, channel)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """

        data = (
            customer_id,
            result['cluster'],
            result['churn']['proba_churn'],
            result['churn']['pred_churn'],
            result['churn']['churn_risk_pct'],
            result['expected_loss']['ExpectedLoss_real'],
            result['recommendation']['action_id'],
            result['recommendation']['action_name'],
            result['recommendation']['channel']
        )

        cur.execute(sql, data)
        conn.commit()
        conn.close()

        print("✓ Lịch sử đã được lưu vào MySQL")

    except Exception as e:
        print("Lỗi khi lưu lịch sử:", e)
