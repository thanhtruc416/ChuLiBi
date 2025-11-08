CREATE DATABASE  IF NOT EXISTS `chulibi` /*!40100 DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci */ /*!80016 DEFAULT ENCRYPTION='N' */;
USE `chulibi`;
-- MySQL dump 10.13  Distrib 8.0.43, for Win64 (x86_64)
--
-- Host: 127.0.0.1    Database: chulibi
-- ------------------------------------------------------
-- Server version	8.0.43

/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!50503 SET NAMES utf8 */;
/*!40103 SET @OLD_TIME_ZONE=@@TIME_ZONE */;
/*!40103 SET TIME_ZONE='+00:00' */;
/*!40014 SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;

--
-- Table structure for table `password_resets`
--

DROP TABLE IF EXISTS `password_resets`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `password_resets` (
  `id` int NOT NULL AUTO_INCREMENT,
  `user_id` int DEFAULT NULL,
  `email` varchar(255) DEFAULT NULL,
  `otp_hash` varchar(255) NOT NULL,
  `expires_at` datetime NOT NULL,
  `used` tinyint(1) NOT NULL DEFAULT '0',
  `created_at` timestamp NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  KEY `user_id` (`user_id`),
  KEY `idx_pr_email_used_exp` (`email`,`used`,`expires_at`),
  KEY `idx_pr_user_used_exp` (`user_id`,`used`,`expires_at`),
  CONSTRAINT `fk_password_resets_user` FOREIGN KEY (`user_id`) REFERENCES `user_data` (`id`) ON DELETE CASCADE
) ENGINE=InnoDB AUTO_INCREMENT=34 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `password_resets`
--

LOCK TABLES `password_resets` WRITE;
/*!40000 ALTER TABLE `password_resets` DISABLE KEYS */;
INSERT INTO `password_resets` VALUES (1,1,NULL,'$2b$12$S0n5plqHGZBRVRFvcuF5xOnUxYZNUNfCIFgi4koA6R0K7mdfBfYGa','2025-10-11 16:01:53',0,'2025-10-11 15:51:53'),(2,1,NULL,'$2b$12$/Al1W33ZLMeB0TJ.AVxRcOACiL/LiB8Km0fatc1p6huBxkZS6J3s.','2025-10-11 16:04:14',0,'2025-10-11 15:54:14'),(3,1,NULL,'$2b$12$NaDnPWlVfKk0ilyzwaMKheHm2ExaNzN387Eba0QjbebfSlV5z/grC','2025-10-11 16:16:13',1,'2025-10-11 16:06:13'),(4,1,NULL,'$2b$12$j216AbkSNT/wO76a1q.W3OMRHEyww7am47g/l3QXZW5VwXYUGkwiS','2025-10-11 17:36:01',0,'2025-10-11 17:26:01'),(5,1,NULL,'$2b$12$qQQAjqBrJNxWXKoiVgh1P.dBRQWtcXG.KeLow2aoYb/2n/Xh3caFm','2025-10-11 17:39:08',0,'2025-10-11 17:29:08'),(6,1,NULL,'$2b$12$rFuJJqAGPviEUyY.9mWf..3wbg.uEC6JWDOLO2T8OPXgjh8XiqSZi','2025-10-11 17:42:45',0,'2025-10-11 17:32:45'),(7,1,NULL,'$2b$12$WxTaWweg.BKJikYIRuA/be3ffkzRYToyioWMMPcGmtGK/ac9eUfD.','2025-10-11 17:43:43',0,'2025-10-11 17:33:43'),(8,1,NULL,'$2b$12$XWYp0ColnbUDjuYvd/q3A.Rt/U9OqUcCv/CD77osSiGaDYkb8HhBa','2025-10-11 17:51:32',0,'2025-10-11 17:41:32'),(9,1,NULL,'$2b$12$0mW9PExmJxHecd2xQJrTo.GuP1qAyrAgkHZokLbGWyjg4FBfUiAn.','2025-10-11 17:52:42',0,'2025-10-11 17:42:42'),(10,1,NULL,'$2b$12$YajDMIMuop0uxaChpOzu2.7AHs/F1zfOr9lElkkLkoE/3qoNO5rAi','2025-10-11 17:56:38',0,'2025-10-11 17:46:38'),(11,1,NULL,'$2b$12$IZD/bMQ/Ef6izWgM207db.cKbokWVbGc9VorWdqSpWaMDDAcPPx9m','2025-10-11 18:09:56',0,'2025-10-11 17:59:56'),(12,1,NULL,'$2b$12$siu4ZHjZbu9VLRGaxivUSuw7xawuLG8uy2V4QsbaOqGkzchR9v1sq','2025-10-14 00:28:14',0,'2025-10-14 00:18:14'),(13,1,NULL,'$2b$12$mx0LasxTmSwCxDEepC3BxeZE0tYtRV0chOFiEqzNiaidzFaIpAQLW','2025-10-16 08:48:49',0,'2025-10-16 08:38:49'),(14,1,NULL,'$2b$12$D/PclEjm0YnXzdoGk7m/Z.CjNQfETPOlaJRRERseyf/4xsYvglhoa','2025-10-16 08:50:07',0,'2025-10-16 08:40:07'),(15,1,NULL,'$2b$12$USzAfDZax4WMBuAUTCzsLuvJro57oweaXDAhvgo1hjQIJakmRn/Ki','2025-10-23 12:03:26',0,'2025-10-23 11:53:26'),(16,1,NULL,'$2b$12$Cy7/7VcdHk2cJr5lYFfzv.xUbpSm7DGK1Sz4D6OHEtAOcUTN4GQYe','2025-10-23 12:07:31',0,'2025-10-23 11:57:31'),(17,1,NULL,'$2b$12$V6qrT60LUlPMsyvI43NGaOuiP0mVU6evTdsizMi5VmSdmzKjwXUEa','2025-10-23 12:08:57',0,'2025-10-23 11:58:57'),(18,1,NULL,'$2b$12$IV5b7G6z0VQfMfEPfCIxbuQ2GTu8tJZxnuMldiZCCTbcZx.BeEvqe','2025-10-23 12:10:02',0,'2025-10-23 12:00:02'),(19,1,NULL,'$2b$12$gA0ZuszuqxlGa.UdR.jGg.AKx/n6EykpjKJFlD6B6XHSW25brG75O','2025-10-23 12:11:10',0,'2025-10-23 12:01:10'),(20,NULL,'icecute2k3@gmail.com','$2b$12$lhmsqm8XKrIU2Kzlu9NiX.f1VpPi9lKvnnKUWb5tmAqo2AsABJowq','2025-10-28 18:21:44',0,'2025-10-28 18:11:44'),(21,NULL,'icecute2k3@gmail.com','$2b$12$ski9zhAd.04b8Sj6Ynub0.avq6fQVNiBVUgdBijBpEmttMjaHBh2e','2025-10-28 18:44:16',0,'2025-10-28 18:34:16'),(22,NULL,'icecute2k3@gmail.com','$2b$12$O/nUzjkdx1/dottFIMacwu0hpZnkg2SB8Q/InadHoJ8tM5p.Ir09C','2025-10-28 18:46:06',0,'2025-10-28 18:36:06'),(23,NULL,'icecute2k3@gmail.com','$2b$12$c8nF.H0Pgtf9pulZjUMzZ.xDz0nVMysQ1kWnCcN2AUaM.s4UqrSIK','2025-10-28 18:46:23',0,'2025-10-28 18:36:23'),(24,NULL,'icecute2k3@gmail.com','$2b$12$bxhJfvnLpXt8leyfzw1HfOUgjNqYnpssHU6lAoOBWi2ZAxnx1MGnu','2025-10-29 15:39:33',0,'2025-10-29 15:29:33'),(25,NULL,'icecute2k3@gmail.com','$2b$12$TLpwdZ9Wys5CFRjPFy6.N.nhfUt/KqQ1IcoyZ.z3jk2DOLLR2S3qy','2025-10-29 15:53:42',0,'2025-10-29 08:43:42'),(26,NULL,'icecute2k3@gmail.com','$2b$12$YjauOmvBwT60KPbm/fKDoeRvoFkSLZlHGjeNq2Pywl3tWbrHcFe7O','2025-10-29 16:02:05',0,'2025-10-29 08:52:05'),(27,NULL,'icecute2k3@gmail.com','$2b$12$CNruOOZiCHHQZd.VfIhkp.0SZM517V0GB0wYbgH3dGncpCqurqc6K','2025-10-29 16:03:50',0,'2025-10-29 08:53:50'),(28,NULL,'icecute2k3@gmail.com','$2b$12$1PXExFDzP2nrQs6vF0hSv.C/BMfqANbt5xaVZwZCbYinKm1BuDici','2025-10-29 16:14:44',0,'2025-10-29 09:04:44'),(29,NULL,'icecute2k3@gmail.com','$2b$12$M/UHHEuhrfs4M9lHKRFWte.MXAx3GS9Erhkgzr.DADFRh4vsJW2Gy','2025-10-29 16:23:55',1,'2025-10-29 09:13:55'),(30,NULL,'icecute2k3@gmail.com','$2b$12$EzPKrYF.I2s6/oxVOx0D9.y3y7U.KoocStxRaMBfSm7.mFunwtMz6','2025-10-29 16:30:50',1,'2025-10-29 09:20:50'),(31,NULL,'icecute2k3@gmail.com','$2b$12$H/6mFmERwUSkg2vNFnOP0uyTMsvb1GD.qZ12LBPfvsXd410oum7i.','2025-10-29 16:38:07',0,'2025-10-29 09:28:07'),(32,NULL,'icecute2k3@gmail.com','$2b$12$RqxG6IF8b2lQIRxrYyMM/eh.jTZQAFBIxDto3DmHOJ3DNepKAY5eW','2025-11-02 15:26:24',1,'2025-11-02 08:16:24'),(33,NULL,'chinnq23416@st.uel.edu.vn','$2b$12$9XFrKBvsgf76I62VM82TwOKd2lQY9TjcXlE7csElPQmyaqwbFQLtK','2025-11-02 15:42:52',0,'2025-11-02 08:32:52');
/*!40000 ALTER TABLE `password_resets` ENABLE KEYS */;
UNLOCK TABLES;
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2025-11-02 23:02:43
