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
) ENGINE=InnoDB AUTO_INCREMENT=58 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `password_resets`
--

LOCK TABLES `password_resets` WRITE;
/*!40000 ALTER TABLE `password_resets` DISABLE KEYS */;
INSERT INTO `password_resets` VALUES (20,NULL,'icecute2k3@gmail.com','$2b$12$lhmsqm8XKrIU2Kzlu9NiX.f1VpPi9lKvnnKUWb5tmAqo2AsABJowq','2025-10-28 18:21:44',0,'2025-10-28 18:11:44'),(21,NULL,'icecute2k3@gmail.com','$2b$12$ski9zhAd.04b8Sj6Ynub0.avq6fQVNiBVUgdBijBpEmttMjaHBh2e','2025-10-28 18:44:16',0,'2025-10-28 18:34:16'),(22,NULL,'icecute2k3@gmail.com','$2b$12$O/nUzjkdx1/dottFIMacwu0hpZnkg2SB8Q/InadHoJ8tM5p.Ir09C','2025-10-28 18:46:06',0,'2025-10-28 18:36:06'),(23,NULL,'icecute2k3@gmail.com','$2b$12$c8nF.H0Pgtf9pulZjUMzZ.xDz0nVMysQ1kWnCcN2AUaM.s4UqrSIK','2025-10-28 18:46:23',0,'2025-10-28 18:36:23'),(24,NULL,'icecute2k3@gmail.com','$2b$12$bxhJfvnLpXt8leyfzw1HfOUgjNqYnpssHU6lAoOBWi2ZAxnx1MGnu','2025-10-29 15:39:33',0,'2025-10-29 15:29:33'),(25,NULL,'icecute2k3@gmail.com','$2b$12$TLpwdZ9Wys5CFRjPFy6.N.nhfUt/KqQ1IcoyZ.z3jk2DOLLR2S3qy','2025-10-29 15:53:42',0,'2025-10-29 08:43:42'),(26,NULL,'icecute2k3@gmail.com','$2b$12$YjauOmvBwT60KPbm/fKDoeRvoFkSLZlHGjeNq2Pywl3tWbrHcFe7O','2025-10-29 16:02:05',0,'2025-10-29 08:52:05'),(27,NULL,'icecute2k3@gmail.com','$2b$12$CNruOOZiCHHQZd.VfIhkp.0SZM517V0GB0wYbgH3dGncpCqurqc6K','2025-10-29 16:03:50',0,'2025-10-29 08:53:50'),(28,NULL,'icecute2k3@gmail.com','$2b$12$1PXExFDzP2nrQs6vF0hSv.C/BMfqANbt5xaVZwZCbYinKm1BuDici','2025-10-29 16:14:44',0,'2025-10-29 09:04:44'),(29,NULL,'icecute2k3@gmail.com','$2b$12$M/UHHEuhrfs4M9lHKRFWte.MXAx3GS9Erhkgzr.DADFRh4vsJW2Gy','2025-10-29 16:23:55',1,'2025-10-29 09:13:55'),(30,NULL,'icecute2k3@gmail.com','$2b$12$EzPKrYF.I2s6/oxVOx0D9.y3y7U.KoocStxRaMBfSm7.mFunwtMz6','2025-10-29 16:30:50',1,'2025-10-29 09:20:50'),(31,NULL,'icecute2k3@gmail.com','$2b$12$H/6mFmERwUSkg2vNFnOP0uyTMsvb1GD.qZ12LBPfvsXd410oum7i.','2025-10-29 16:38:07',0,'2025-10-29 09:28:07'),(32,NULL,'icecute2k3@gmail.com','$2b$12$RqxG6IF8b2lQIRxrYyMM/eh.jTZQAFBIxDto3DmHOJ3DNepKAY5eW','2025-11-02 15:26:24',1,'2025-11-02 08:16:24'),(33,NULL,'chinnq23416@st.uel.edu.vn','$2b$12$9XFrKBvsgf76I62VM82TwOKd2lQY9TjcXlE7csElPQmyaqwbFQLtK','2025-11-02 15:42:52',0,'2025-11-02 08:32:52'),(34,NULL,'lqd.tk22.caothithanhtruc@gmail.com','$2b$12$UbDHq6ir1HNoo/Bp/2BZnuQD6t3MWl3QL7yN1SXUdXQa0pHrQ6Vgm','2025-11-04 02:27:07',0,'2025-11-03 19:17:07'),(35,NULL,'anhnth23416@uel.edu.vn','$2b$12$GdXhP6IcuJ4iW9jVjerevuJw8OE0tyHD25Zgv1ZtCYhUOLaIZzcGi','2025-11-04 02:27:59',0,'2025-11-03 19:17:59'),(36,NULL,'anhnth23416@st.uel.edu.vn','$2b$12$c2qyRa4YlGc2ve6kEVLg5u0QnGEh3Fg4/Rl0qmp06N2I44SLVU1n2','2025-11-04 02:29:25',1,'2025-11-03 19:19:25'),(44,NULL,'lqd.tik22.caothithanhtruc@gmail.com','$2b$12$3phthLKHWtpq8xvbYlIf/OyQRGBmr3mj.eBCPpetQPFmNaAFikuXW','2025-11-08 15:43:16',0,'2025-11-08 08:33:16'),(45,NULL,'lqd.tik22.caothithanhtruc@gmail.com','$2b$12$qqY/AyG8G4t/077wyFJjqOKt5S3crybfvOq4OJC1CRwuMaQvC/nBm','2025-11-08 15:44:27',0,'2025-11-08 08:34:27'),(46,NULL,'lqd.tik22.caothithanhtruc@gmail.com','$2b$12$L6fkAAH.J1yz.QPVLh4./edS/1luy6DpjZ4QPZl/3ej/nYgYRjhJe','2025-11-08 15:46:14',0,'2025-11-08 08:36:14'),(47,NULL,'icecute2k3@gmail.com','$2b$12$P4DrXB8RWmhLIEuguaaDjeEeClBPokDDPBU36qDMvC1AdnV47NG2K','2025-11-08 15:50:02',1,'2025-11-08 08:40:02'),(49,NULL,'trucctt23416@st.uel.edu.vn','$2b$12$oiPDksGUUTC1n/PYutFcsuXuEyBq.rwWN3rjb2paft59qxBnWZb6m','2025-11-08 15:59:10',0,'2025-11-08 08:49:10'),(51,NULL,'thanhtruc.yee@gmail.com','$2b$12$xQNDIPnJEmmE8IuYae8zdus8IjdLvYXkzyRZjUDmuX3s/dO0y8.t2','2025-11-10 15:58:14',1,'2025-11-10 08:48:14'),(52,NULL,'trucctt23416@st.uel.edu.vn','$2b$12$4RvmowwDm0EHSy5KUgFWzOzqqeTZehux2R6973gWM38ZJUpDbtlLW','2025-11-13 04:52:24',1,'2025-11-12 21:42:24'),(54,NULL,'trucctt23416@st.uel.edu.vn','$2b$12$Ux2fDVg3TDngCgYwp5iZ7uZ7IcE6BBKtGVp48OlBaGDvO27wdKDMS','2025-11-13 07:34:12',1,'2025-11-13 00:24:12'),(55,11,NULL,'$2b$12$e0b.7ZNcOVAJ/bEhZHRpcek24tcgp9afmdTXkx0xWh9RhzsOdVsXS','2025-11-13 07:35:24',1,'2025-11-13 07:25:24'),(56,11,NULL,'$2b$12$KV37tvew015ZBlcR5OSie.8AXikBuktoq5EbAJl9vo/R/gCp2mj9.','2025-11-16 13:10:51',1,'2025-11-16 13:00:51'),(57,9,NULL,'$2b$12$bsOujlO.XlCkoNiXqHrnxu2Y2nmTeRmYh4dNX2gRo3r0uEte1lVOO','2025-11-16 13:30:32',1,'2025-11-16 13:20:32');
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

-- Dump completed on 2025-11-16 20:51:43
