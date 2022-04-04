-- phpMyAdmin SQL Dump
-- version 4.0.4
-- http://www.phpmyadmin.net
--
-- Host: localhost
-- Generation Time: Feb 03, 2022 at 07:11 AM
-- Server version: 5.6.12-log
-- PHP Version: 5.4.16

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8 */;

--
-- Database: `urban_rail_transit_systems`
--
CREATE DATABASE IF NOT EXISTS `urban_rail_transit_systems` DEFAULT CHARACTER SET latin1 COLLATE latin1_swedish_ci;
USE `urban_rail_transit_systems`;

-- --------------------------------------------------------

--
-- Table structure for table `auth_group`
--

CREATE TABLE IF NOT EXISTS `auth_group` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `name` varchar(80) NOT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `name` (`name`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1 AUTO_INCREMENT=1 ;

-- --------------------------------------------------------

--
-- Table structure for table `auth_group_permissions`
--

CREATE TABLE IF NOT EXISTS `auth_group_permissions` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `group_id` int(11) NOT NULL,
  `permission_id` int(11) NOT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `auth_group_permissions_group_id_permission_id_0cd325b0_uniq` (`group_id`,`permission_id`),
  KEY `auth_group_permissio_permission_id_84c5c92e_fk_auth_perm` (`permission_id`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1 AUTO_INCREMENT=1 ;

-- --------------------------------------------------------

--
-- Table structure for table `auth_permission`
--

CREATE TABLE IF NOT EXISTS `auth_permission` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `name` varchar(255) NOT NULL,
  `content_type_id` int(11) NOT NULL,
  `codename` varchar(100) NOT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `auth_permission_content_type_id_codename_01ab375a_uniq` (`content_type_id`,`codename`)
) ENGINE=InnoDB  DEFAULT CHARSET=latin1 AUTO_INCREMENT=28 ;

--
-- Dumping data for table `auth_permission`
--

INSERT INTO `auth_permission` (`id`, `name`, `content_type_id`, `codename`) VALUES
(1, 'Can add log entry', 1, 'add_logentry'),
(2, 'Can change log entry', 1, 'change_logentry'),
(3, 'Can delete log entry', 1, 'delete_logentry'),
(4, 'Can add permission', 2, 'add_permission'),
(5, 'Can change permission', 2, 'change_permission'),
(6, 'Can delete permission', 2, 'delete_permission'),
(7, 'Can add group', 3, 'add_group'),
(8, 'Can change group', 3, 'change_group'),
(9, 'Can delete group', 3, 'delete_group'),
(10, 'Can add user', 4, 'add_user'),
(11, 'Can change user', 4, 'change_user'),
(12, 'Can delete user', 4, 'delete_user'),
(13, 'Can add content type', 5, 'add_contenttype'),
(14, 'Can change content type', 5, 'change_contenttype'),
(15, 'Can delete content type', 5, 'delete_contenttype'),
(16, 'Can add session', 6, 'add_session'),
(17, 'Can change session', 6, 'change_session'),
(18, 'Can delete session', 6, 'delete_session'),
(19, 'Can add client register_ model', 7, 'add_clientregister_model'),
(20, 'Can change client register_ model', 7, 'change_clientregister_model'),
(21, 'Can delete client register_ model', 7, 'delete_clientregister_model'),
(22, 'Can add client posts_ model', 8, 'add_clientposts_model'),
(23, 'Can change client posts_ model', 8, 'change_clientposts_model'),
(24, 'Can delete client posts_ model', 8, 'delete_clientposts_model'),
(25, 'Can add feedbacks_ model', 9, 'add_feedbacks_model'),
(26, 'Can change feedbacks_ model', 9, 'change_feedbacks_model'),
(27, 'Can delete feedbacks_ model', 9, 'delete_feedbacks_model');

-- --------------------------------------------------------

--
-- Table structure for table `auth_user`
--

CREATE TABLE IF NOT EXISTS `auth_user` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `password` varchar(128) NOT NULL,
  `last_login` datetime(6) DEFAULT NULL,
  `is_superuser` tinyint(1) NOT NULL,
  `username` varchar(150) NOT NULL,
  `first_name` varchar(30) NOT NULL,
  `last_name` varchar(150) NOT NULL,
  `email` varchar(254) NOT NULL,
  `is_staff` tinyint(1) NOT NULL,
  `is_active` tinyint(1) NOT NULL,
  `date_joined` datetime(6) NOT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `username` (`username`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1 AUTO_INCREMENT=1 ;

-- --------------------------------------------------------

--
-- Table structure for table `auth_user_groups`
--

CREATE TABLE IF NOT EXISTS `auth_user_groups` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `user_id` int(11) NOT NULL,
  `group_id` int(11) NOT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `auth_user_groups_user_id_group_id_94350c0c_uniq` (`user_id`,`group_id`),
  KEY `auth_user_groups_group_id_97559544_fk_auth_group_id` (`group_id`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1 AUTO_INCREMENT=1 ;

-- --------------------------------------------------------

--
-- Table structure for table `auth_user_user_permissions`
--

CREATE TABLE IF NOT EXISTS `auth_user_user_permissions` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `user_id` int(11) NOT NULL,
  `permission_id` int(11) NOT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `auth_user_user_permissions_user_id_permission_id_14a6b632_uniq` (`user_id`,`permission_id`),
  KEY `auth_user_user_permi_permission_id_1fbb5f2c_fk_auth_perm` (`permission_id`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1 AUTO_INCREMENT=1 ;

-- --------------------------------------------------------

--
-- Table structure for table `django_admin_log`
--

CREATE TABLE IF NOT EXISTS `django_admin_log` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `action_time` datetime(6) NOT NULL,
  `object_id` longtext,
  `object_repr` varchar(200) NOT NULL,
  `action_flag` smallint(5) unsigned NOT NULL,
  `change_message` longtext NOT NULL,
  `content_type_id` int(11) DEFAULT NULL,
  `user_id` int(11) NOT NULL,
  PRIMARY KEY (`id`),
  KEY `django_admin_log_content_type_id_c4bce8eb_fk_django_co` (`content_type_id`),
  KEY `django_admin_log_user_id_c564eba6_fk_auth_user_id` (`user_id`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1 AUTO_INCREMENT=1 ;

-- --------------------------------------------------------

--
-- Table structure for table `django_content_type`
--

CREATE TABLE IF NOT EXISTS `django_content_type` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `app_label` varchar(100) NOT NULL,
  `model` varchar(100) NOT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `django_content_type_app_label_model_76bd3d3b_uniq` (`app_label`,`model`)
) ENGINE=InnoDB  DEFAULT CHARSET=latin1 AUTO_INCREMENT=10 ;

--
-- Dumping data for table `django_content_type`
--

INSERT INTO `django_content_type` (`id`, `app_label`, `model`) VALUES
(1, 'admin', 'logentry'),
(3, 'auth', 'group'),
(2, 'auth', 'permission'),
(4, 'auth', 'user'),
(8, 'Client_Site', 'clientposts_model'),
(7, 'Client_Site', 'clientregister_model'),
(9, 'Client_Site', 'feedbacks_model'),
(5, 'contenttypes', 'contenttype'),
(6, 'sessions', 'session');

-- --------------------------------------------------------

--
-- Table structure for table `django_migrations`
--

CREATE TABLE IF NOT EXISTS `django_migrations` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `app` varchar(255) NOT NULL,
  `name` varchar(255) NOT NULL,
  `applied` datetime(6) NOT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB  DEFAULT CHARSET=latin1 AUTO_INCREMENT=22 ;

--
-- Dumping data for table `django_migrations`
--

INSERT INTO `django_migrations` (`id`, `app`, `name`, `applied`) VALUES
(1, 'Remote_User', '0001_initial', '2019-04-23 07:01:48.050781'),
(2, 'contenttypes', '0001_initial', '2019-04-23 07:01:49.494140'),
(3, 'auth', '0001_initial', '2019-04-23 07:02:03.837890'),
(4, 'admin', '0001_initial', '2019-04-23 07:02:05.832031'),
(5, 'admin', '0002_logentry_remove_auto_add', '2019-04-23 07:02:05.863281'),
(6, 'contenttypes', '0002_remove_content_type_name', '2019-04-23 07:02:07.041015'),
(7, 'auth', '0002_alter_permission_name_max_length', '2019-04-23 07:02:07.839843'),
(8, 'auth', '0003_alter_user_email_max_length', '2019-04-23 07:02:08.330078'),
(9, 'auth', '0004_alter_user_username_opts', '2019-04-23 07:02:08.361328'),
(10, 'auth', '0005_alter_user_last_login_null', '2019-04-23 07:02:08.921875'),
(11, 'auth', '0006_require_contenttypes_0002', '2019-04-23 07:02:08.953125'),
(12, 'auth', '0007_alter_validators_add_error_messages', '2019-04-23 07:02:08.989257'),
(13, 'auth', '0008_alter_user_username_max_length', '2019-04-23 07:02:09.785156'),
(14, 'auth', '0009_alter_user_last_name_max_length', '2019-04-23 07:02:10.580078'),
(15, 'sessions', '0001_initial', '2019-04-23 07:02:11.764648'),
(16, 'Remote_User', '0002_clientposts_model', '2019-04-25 05:53:57.702132'),
(17, 'Remote_User', '0003_clientposts_model_usefulcounts', '2019-04-25 10:00:02.521468'),
(18, 'Remote_User', '0004_auto_20190429_1027', '2019-04-29 04:57:32.672296'),
(19, 'Remote_User', '0005_clientposts_model_dislikes', '2019-04-29 05:15:16.668390'),
(20, 'Remote_User', '0006_Review_model', '2019-04-29 05:19:26.382257'),
(21, 'Remote_User', '0007_clientposts_model_names', '2019-04-30 04:45:46.472656');

-- --------------------------------------------------------

--
-- Table structure for table `django_session`
--

CREATE TABLE IF NOT EXISTS `django_session` (
  `session_key` varchar(40) NOT NULL,
  `session_data` longtext NOT NULL,
  `expire_date` datetime(6) NOT NULL,
  PRIMARY KEY (`session_key`),
  KEY `django_session_expire_date_a5c62663` (`expire_date`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Dumping data for table `django_session`
--

INSERT INTO `django_session` (`session_key`, `session_data`, `expire_date`) VALUES
('0jpcgnd1gmwbp3e8tw54e6nxjylsogyo', 'YmM4NjE0MDQ2MzBmYWIxNzIzNTkxZjBiN2I5M2MxMzQyYTE0YmMxODp7InVzZXJpZCI6Mn0=', '2020-02-21 08:52:28.687500'),
('3x40icxutxpcspr42y9bugog9ucvrich', 'eyJ1c2VyaWQiOjE3fQ:1nFW81:tlFoVjA1TZF-P0qE2PFUoy8M-0S1hnTQJkbZ7oN5bRs', '2022-02-17 07:02:01.447350'),
('49qo7iki5uxczhyymi8ka7dnh6a2wva5', 'MmE4N2EzZmM3NTI1ODc3MjUxYjUxNWM3OWM4ZGExNWViMzRkN2MzYTp7Im5hbWUiOjF9', '2019-05-08 09:19:45.486328'),
('4df7s82pddaszour6twx23d86058ppjq', 'ZmNkODA5MmI1ZGQ0Yjk5MmZlNzEyNTcwNTcxNjk2ZWYxZTE3NThkMjp7InVzZXJpZCI6NX0=', '2020-11-23 11:49:21.396484'),
('4io28d085qjfib7a5s2qbhc8qp4wfiva', 'eyJ1c2VyaWQiOjE2fQ:1mAtmi:oIUbcN3WzJiaWnxMBZ6eIGMTo8NS2y701JlpwqvzBUk', '2021-08-17 12:44:40.453750'),
('4x6b78w9rfcn34v650kd2j7oij6atr8p', 'Zjk0Y2RlYjc4OTJhNWMyZjQyNmM4ZGRhYTVjNmVlNDFhZGE4ZmU3NTp7InVzZXJpZCI6Nn0=', '2019-12-27 12:07:42.082031'),
('b9cu6cjsfqfm5mame5dy1ikpiiy7yn3w', 'OTk3NTk2YTE0NjM5MWQ0OGQ0MjY3NzBjNzdhOTc0ZWJhM2ZkMzdkMjp7InVzZXJpZCI6MX0=', '2019-05-09 11:00:08.480453'),
('ct13q5fpn94zvnij8ekixwzcky2imc5e', 'YWUzM2IzMWJiYmQ3YmY2YzlkMGFlNTM1YmU5ZGM4YjQ0MmY1YTc0NTp7InVzZXJpZCI6NH0=', '2019-05-14 11:44:10.978515'),
('e07j4duysh402dedtomm8icctvs9ljgy', 'MmE4N2EzZmM3NTI1ODc3MjUxYjUxNWM3OWM4ZGExNWViMzRkN2MzYTp7Im5hbWUiOjF9', '2019-05-09 06:08:12.306625'),
('hbv74sg6w6e4wp89vq807vw0xhkh5s1h', 'MzU0ZWYzNTQ3MjM4MWZlOTVjM2M1MWQ4MmE5ODE0OTlkNDRkNDkwMDp7InVzZXJpZCI6MX0=', '2020-01-10 07:40:38.067382'),
('hhtt48je70l9nzw6dee4ocuxxm9blqej', 'NGRhY2JkNmQ4ZTM4OTU0Y2UzMzFlZmZmOTgzYmE0MWVkOThiNjc2NTp7Im5hbWUiOjEsInVzZXJpZCI6MX0=', '2019-05-09 10:12:38.380843'),
('ic3hqykgws5iy6fz5ns6h6f921jbjzmt', 'eyJ1c2VyaWQiOjExfQ:1kywHL:I_tahJ0VJb7myAbMbXpWZu9XrSaAMmduNxGd2x5gtmY', '2021-01-25 12:26:35.043761'),
('iz6wcyx97x1w6mpfc51g1tj72z2xghfn', 'eyJ1c2VyaWQiOjl9:1kwlIp:YKOKMwJARe6w057AKTGY1-GCuRcZAeAbJ0bdQao23wY', '2021-01-19 12:19:07.663490'),
('k7dyn4irgrj5wb4jucb4po527iw724dp', 'eyJ1c2VyaWQiOjEzfQ:1l0JrY:2_TJ4L_XoHdOW51Zdp0MOdyBEZEzntk5pdXZFDmX9x4', '2021-01-29 07:49:40.202848'),
('o7x1vhluuypdfmgv7fmv6nohgfn5ub55', 'NzMyZjlhNzFhZjk2ZGUzZmFiMmIzYjMwNTJkYTg5MDUzNmNlMDk4Mjp7InVzZXJpZCI6MTZ9', '2020-01-02 12:51:55.659179'),
('psdjoq42u7lfqwfodftic5x6z9ij34nk', 'eyJ1c2VyaWQiOjExfQ:1mAXDq:a8YYY1YJU3jPv03qo9-VcrjRHnDWRSqGseiR93n0GVM', '2021-08-16 12:39:10.518259'),
('qnaolidvfx6bu9ra3uyqvkgva7bv92f1', 'OTk3NTk2YTE0NjM5MWQ0OGQ0MjY3NzBjNzdhOTc0ZWJhM2ZkMzdkMjp7InVzZXJpZCI6MX0=', '2019-05-14 05:34:50.069335'),
('sdcvtwp7s5yj8q1lb0mdvlg8nj5wujqo', 'eyJ1c2VyaWQiOjEyfQ:1kzJ3p:0g6nRuJv3TXWVpANqNgbJcrUv96ZU5UQwv3bgqBbL1I', '2021-01-26 12:46:09.538596'),
('tejgl09oettnyva23kqdbns5nfz5g8ug', 'OTk3NTk2YTE0NjM5MWQ0OGQ0MjY3NzBjNzdhOTc0ZWJhM2ZkMzdkMjp7InVzZXJpZCI6MX0=', '2019-05-09 11:19:24.387679'),
('u5icgvq3qt5nthdlv99go3r804ccghbo', 'MmE4N2EzZmM3NTI1ODc3MjUxYjUxNWM3OWM4ZGExNWViMzRkN2MzYTp7Im5hbWUiOjF9', '2019-05-09 06:00:13.573226'),
('ws2o4cq1jbqepe0e9s9v7n4erxatq9ic', 'eyJ1c2VyaWQiOjE1fQ:1l2CgI:SmlpAnZzplZhPTFJ_rkEJstnZRl2CYWyTcah7PHPv-M', '2021-02-03 12:33:50.352453'),
('zega5sz46ivu1tb1o1mtmg3v2ysxog1w', 'eyJ1c2VyaWQiOjh9:1kuVm4:L7RizVvw4EC0IyYCYAIhGjC8lvZol_Z1abqVwdkdKkY', '2021-01-13 07:20:00.767751');

-- --------------------------------------------------------

--
-- Table structure for table `remote_user_clientregister_model`
--

CREATE TABLE IF NOT EXISTS `remote_user_clientregister_model` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `username` varchar(30) NOT NULL,
  `email` varchar(30) NOT NULL,
  `password` varchar(50) NOT NULL,
  `phoneno` varchar(50) NOT NULL,
  `country` varchar(30) NOT NULL,
  `state` varchar(30) NOT NULL,
  `city` varchar(30) NOT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB  DEFAULT CHARSET=latin1 AUTO_INCREMENT=18 ;

--
-- Dumping data for table `remote_user_clientregister_model`
--

INSERT INTO `remote_user_clientregister_model` (`id`, `username`, `email`, `password`, `phoneno`, `country`, `state`, `city`) VALUES
(10, 'Govind', 'Govind.123@gmail.com', 'Govind', '9535866270', 'India', 'Karnataka', 'Bangalore'),
(11, 'Manjunath', 'tmksmanju13@gmail.com', 'Manjunath', '9535866270', 'India', 'Karnataka', 'Bangalore'),
(12, 'tmksmanju', 'tmksmanju13@gmail.com', 'tmksmanju', '9535866271', 'India', 'Karnataka', 'Bangalore'),
(13, 'Arvind', 'Arvind123@gmail.com', 'Arvind', '9535866270', 'India', 'Karnataka', 'Bangalore'),
(14, 'Amar', 'Amar123@gmail.com', 'Amar', '9535866270', 'India', 'Karnataka', 'Bangalore'),
(15, 'Anil', 'Anil123@gmail.com', 'Anil', '9535866270', 'India', 'Karnataka', 'Bangalore'),
(16, 'Abilash', 'Abilash123@gmail.com', 'Abilash', '9535866270', 'India', 'Karnataka', 'Bangalore'),
(17, 'Rajesh', 'Rajesh123@gmail.com', 'Rajesh', '9535866270', 'India', 'Karnataka', 'Bangalore');

-- --------------------------------------------------------

--
-- Table structure for table `remote_user_detection_accuracy`
--

CREATE TABLE IF NOT EXISTS `remote_user_detection_accuracy` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `names` varchar(300) NOT NULL,
  `ratio` varchar(300) NOT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB  DEFAULT CHARSET=latin1 AUTO_INCREMENT=61 ;

--
-- Dumping data for table `remote_user_detection_accuracy`
--

INSERT INTO `remote_user_detection_accuracy` (`id`, `names`, `ratio`) VALUES
(55, 'Naive Bayes', '25.0'),
(56, 'SVM', '50.0'),
(57, 'Logistic Regression', '50.0'),
(58, 'Decision Tree Classifier', '50.0'),
(59, 'SGD Classifier', '50.0'),
(60, 'KNeighborsClassifier', '25.0');

-- --------------------------------------------------------

--
-- Table structure for table `remote_user_impact_ratio_model`
--

CREATE TABLE IF NOT EXISTS `remote_user_impact_ratio_model` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `names` varchar(300) NOT NULL,
  `ratio` varchar(60) DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB  DEFAULT CHARSET=latin1 AUTO_INCREMENT=10 ;

--
-- Dumping data for table `remote_user_impact_ratio_model`
--

INSERT INTO `remote_user_impact_ratio_model` (`id`, `names`, `ratio`) VALUES
(7, 'More Late', '31.25'),
(8, 'Average Late', '50.0'),
(9, 'Less Late', '18.75');

-- --------------------------------------------------------

--
-- Table structure for table `remote_user_rail_delay_model`
--

CREATE TABLE IF NOT EXISTS `remote_user_rail_delay_model` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `names` varchar(300) NOT NULL,
  `rail_name` varchar(300) NOT NULL,
  `rail_type` varchar(300) NOT NULL,
  `departure_place` varchar(300) NOT NULL,
  `destination` varchar(300) NOT NULL,
  `departure_date` varchar(300) NOT NULL,
  `departure_time` varchar(300) NOT NULL,
  `arrival_date` varchar(300) NOT NULL,
  `arrival_time` varchar(300) NOT NULL,
  `distruption_place_name` varchar(300) NOT NULL,
  `distruption_reason` varchar(300) NOT NULL,
  `distruption_time` varchar(300) NOT NULL,
  `actual_arrival_time` varchar(300) NOT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB  DEFAULT CHARSET=latin1 AUTO_INCREMENT=49 ;

--
-- Dumping data for table `remote_user_rail_delay_model`
--

INSERT INTO `remote_user_rail_delay_model` (`id`, `names`, `rail_name`, `rail_type`, `departure_place`, `destination`, `departure_date`, `departure_time`, `arrival_date`, `arrival_time`, `distruption_place_name`, `distruption_reason`, `distruption_time`, `actual_arrival_time`) VALUES
(33, '253-5687624', 'Mysore Express', 'Broad Gauge', 'Bangalore', 'Mysore', '2021-12-05 00:00:00', '13:18:00', '2021-12-05 00:00:00', '16:20:00', 'Mandya', 'Elephant Crossing', '30', '15:50:00'),
(34, '213-7142614', 'Brindavan Express', 'Broad Gauge', 'Bangalore', 'Chennai', '2021-12-05 00:00:00', '10:15:00', '2021-12-05 00:00:00', '14:30:00', 'Vellore', 'Track Problem', '20', '14:10:00'),
(35, '217-8143615', 'Kovai Express', 'Broad Gauge', 'Bangalore', 'Kovai', '2021-12-05 00:00:00', '16:30:00', '13/5/2021', '13:10:00', 'Erode', 'Track Changing', '70', '11:50:00'),
(36, '753-5487694', 'Godavari Express', 'Broad Gauge', 'Bangalore', 'Kodavari', '2021-10-02 00:00:00', '07:30:00', '2021-10-02 00:00:00', '23:10:00', 'Nellore', 'Track Demaged due to Heavy Rain', '40', '22:30:00'),
(37, '892-0928373', 'Brindavan Express', 'Broad Gauge', 'Chennai', 'Mysore', '2021-11-07 00:00:00', '08:40:00', '2021-11-07 00:00:00', '18:30:00', 'Bangalore', 'Train Collpased', '80', '17:10:00'),
(38, '092-9827382', 'Mumboy Express', 'Meter Gauge', 'Bangalore', 'Mumboy', '18/09/2021', '06:40:00', '18/09/2021', '23:10:00', 'Tumkur', 'Train Traffic', '50', '22:20:00'),
(39, '072-9837216', 'Lalbag Express', 'Broad Gauge', 'Bangalore', 'Mysore', '18/03/2021', '09:30:00', '18/03/2021', '14:40:00', 'Mandya', 'Track Broken', '28', '14:12:00'),
(40, '071-3847921', 'Brindavan Express', 'Broad Gauge', 'Bangalore', 'Chennai', '17/02/2021', '10:15:00', '17/02/2021', '14:30:00', 'Jolarpettai', 'Track Problem', '80', '13:00:00'),
(41, '901-9283763', 'Nellore Express', 'Broad Gauge', 'Bangalore', 'Nellore', '2021-12-05 00:00:00', '16:30:00', '2021-12-05 00:00:00', '13:10:00', 'Hydrabad', 'Track Changing', '70', '11:50:00'),
(42, '892-8673652', 'Tiruathi Express', 'Broad Gauge', 'Bangalore', 'Tirupathi', '2021-12-05 00:00:00', '13:18:00', '2021-12-05 00:00:00', '16:20:00', 'Renigunda', 'Elephant Crossing', '30', '15:50:00'),
(43, '178-2983721', 'Guntur Express', 'Broad Gauge', 'Bangalore', 'Guntur', '16/07/2021', '09:10:00', '16/07/2021', '19:35:00', 'Tirupathi', 'Track Problem', '35', '19:30:00'),
(44, '928-8738272', 'Prashanth Express', 'Broad Gauge', 'Bangalore', 'Guntur', '2021-12-05 00:00:00', '16:30:00', '13/5/2021', '13:10:00', 'Nellore', 'Track Changing', '70', '11:50:00'),
(45, '982-0289371', 'Vizaq Express', 'Broad Gauge', 'Bangalore', 'Visakapatnam', '21/09/2020', '01:10:00', '21/09/2021', '23:10:00', 'Guntur', 'Elephant Crossing', '20', '22:50:00'),
(46, '213-7142614', 'Rajahmundry Express', 'Broad Gauge', 'Bangalore', 'Rajahmundry', '2021-01-03 00:00:00', '02:10:00', '2021-01-03 00:00:00', '23:45:00', 'Nellore', 'Track Problem', '35', '23:10:00'),
(47, '938-0298321', 'Salem Express', 'Meter Gauge', 'Bangalore', 'Salem', '13/06/2021', '05:30:00', '13/06/2021', '11:45:00', 'Dharmapuri', 'Train Accident', '40', '11:05:00'),
(48, '092-9837371', 'Dubble Tucker', 'Broad Gauge', 'Bangalore', 'Chennai', '2021-12-05 00:00:00', '06:10:00', '2021-12-05 00:00:00', '13:30:00', 'Kuppam', 'Elephant Crossing', '30', '13:00:00');

-- --------------------------------------------------------

--
-- Table structure for table `remote_user_rail_delay_prediction_model`
--

CREATE TABLE IF NOT EXISTS `remote_user_rail_delay_prediction_model` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `names` varchar(300) NOT NULL,
  `rail_name` varchar(300) NOT NULL,
  `rail_type` varchar(300) NOT NULL,
  `departure_place` varchar(300) NOT NULL,
  `destination` varchar(300) NOT NULL,
  `departure_date` varchar(300) NOT NULL,
  `departure_time` varchar(300) NOT NULL,
  `arrival_date` varchar(300) NOT NULL,
  `arrival_time` varchar(300) NOT NULL,
  `distruption_place_name` varchar(300) NOT NULL,
  `distruption_reason` varchar(300) NOT NULL,
  `distruption_time` varchar(300) NOT NULL,
  `actual_arrival_time` varchar(300) NOT NULL,
  `impact` varchar(300) NOT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB  DEFAULT CHARSET=latin1 AUTO_INCREMENT=177 ;

--
-- Dumping data for table `remote_user_rail_delay_prediction_model`
--

INSERT INTO `remote_user_rail_delay_prediction_model` (`id`, `names`, `rail_name`, `rail_type`, `departure_place`, `destination`, `departure_date`, `departure_time`, `arrival_date`, `arrival_time`, `distruption_place_name`, `distruption_reason`, `distruption_time`, `actual_arrival_time`, `impact`) VALUES
(161, '253-5687624', 'Mysore Express', 'Broad Gauge', 'Bangalore', 'Mysore', '2021-12-05 00:00:00', '13:18:00', '2021-12-05 00:00:00', '16:20:00', 'Mandya', 'Elephant Crossing', '30', '15:50:00', 'Average Late'),
(162, '213-7142614', 'Brindavan Express', 'Broad Gauge', 'Bangalore', 'Chennai', '2021-12-05 00:00:00', '10:15:00', '2021-12-05 00:00:00', '14:30:00', 'Vellore', 'Track Problem', '20', '14:10:00', 'Less Late'),
(163, '217-8143615', 'Kovai Express', 'Broad Gauge', 'Bangalore', 'Kovai', '2021-12-05 00:00:00', '16:30:00', '13/5/2021', '13:10:00', 'Erode', 'Track Changing', '70', '11:50:00', 'More Late'),
(164, '753-5487694', 'Godavari Express', 'Broad Gauge', 'Bangalore', 'Kodavari', '2021-10-02 00:00:00', '07:30:00', '2021-10-02 00:00:00', '23:10:00', 'Nellore', 'Track Demaged due to Heavy Rain', '40', '22:30:00', 'Average Late'),
(165, '892-0928373', 'Brindavan Express', 'Broad Gauge', 'Chennai', 'Mysore', '2021-11-07 00:00:00', '08:40:00', '2021-11-07 00:00:00', '18:30:00', 'Bangalore', 'Train Collpased', '80', '17:10:00', 'More Late'),
(166, '092-9827382', 'Mumboy Express', 'Meter Gauge', 'Bangalore', 'Mumboy', '18/09/2021', '06:40:00', '18/09/2021', '23:10:00', 'Tumkur', 'Train Traffic', '50', '22:20:00', 'Average Late'),
(167, '072-9837216', 'Lalbag Express', 'Broad Gauge', 'Bangalore', 'Mysore', '18/03/2021', '09:30:00', '18/03/2021', '14:40:00', 'Mandya', 'Track Broken', '28', '14:12:00', 'Less Late'),
(168, '071-3847921', 'Brindavan Express', 'Broad Gauge', 'Bangalore', 'Chennai', '17/02/2021', '10:15:00', '17/02/2021', '14:30:00', 'Jolarpettai', 'Track Problem', '80', '13:00:00', 'More Late'),
(169, '901-9283763', 'Nellore Express', 'Broad Gauge', 'Bangalore', 'Nellore', '2021-12-05 00:00:00', '16:30:00', '2021-12-05 00:00:00', '13:10:00', 'Hydrabad', 'Track Changing', '70', '11:50:00', 'More Late'),
(170, '892-8673652', 'Tiruathi Express', 'Broad Gauge', 'Bangalore', 'Tirupathi', '2021-12-05 00:00:00', '13:18:00', '2021-12-05 00:00:00', '16:20:00', 'Renigunda', 'Elephant Crossing', '30', '15:50:00', 'Average Late'),
(171, '178-2983721', 'Guntur Express', 'Broad Gauge', 'Bangalore', 'Guntur', '16/07/2021', '09:10:00', '16/07/2021', '19:35:00', 'Tirupathi', 'Track Problem', '35', '19:30:00', 'Average Late'),
(172, '928-8738272', 'Prashanth Express', 'Broad Gauge', 'Bangalore', 'Guntur', '2021-12-05 00:00:00', '16:30:00', '13/5/2021', '13:10:00', 'Nellore', 'Track Changing', '70', '11:50:00', 'More Late'),
(173, '982-0289371', 'Vizaq Express', 'Broad Gauge', 'Bangalore', 'Visakapatnam', '21/09/2020', '01:10:00', '21/09/2021', '23:10:00', 'Guntur', 'Elephant Crossing', '20', '22:50:00', 'Less Late'),
(174, '213-7142614', 'Rajahmundry Express', 'Broad Gauge', 'Bangalore', 'Rajahmundry', '2021-01-03 00:00:00', '02:10:00', '2021-01-03 00:00:00', '23:45:00', 'Nellore', 'Track Problem', '35', '23:10:00', 'Average Late'),
(175, '938-0298321', 'Salem Express', 'Meter Gauge', 'Bangalore', 'Salem', '13/06/2021', '05:30:00', '13/06/2021', '11:45:00', 'Dharmapuri', 'Train Accident', '40', '11:05:00', 'Average Late'),
(176, '092-9837371', 'Dubble Tucker', 'Broad Gauge', 'Bangalore', 'Chennai', '2021-12-05 00:00:00', '06:10:00', '2021-12-05 00:00:00', '13:30:00', 'Kuppam', 'Elephant Crossing', '30', '13:00:00', 'Average Late');

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
