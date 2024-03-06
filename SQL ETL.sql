--table creation by dominique miranda (answers at the bottom)
IF NOT EXISTS(SELECT * FROM SYS.DATABASES WHERE NAME = 'DBA_TEST')	
	CREATE DATABASE DBA_TEST
GO

DBA_TEST
GO

--DROP TABLES IF THEY EXIST IN THE DATABASE
IF OBJECT_ID('TBL_TRANSACTIONS', 'U')  IS NOT NULL
	DROP TABLE TBL_TRANSACTIONS;
GO

IF OBJECT_ID('TBL_PHYSICIANS', 'U')  IS NOT NULL
	DROP TABLE TBL_PHYSICIANS
GO

IF OBJECT_ID('TBL_DEPARTMENTS', 'U') IS NOT NULL
	DROP TABLE TBL_DEPARTMENTS
GO

IF OBJECT_ID('TBL_PATIENTS', 'U')  IS NOT NULL
	DROP TABLE TBL_PATIENTS
GO

IF OBJECT_ID('TBL_INSURANCES', 'U')  IS NOT NULL
	DROP TABLE TBL_INSURANCES
GO



--THIS TABLE STORES THE NAMES OF THE DEPARTMENTS/ SPECIALTIES
CREATE TABLE TBL_DEPARTMENTS(
DEPT_ID VARCHAR(10) CONSTRAINT PK_TBL_DEPARTMENTS_DEPT_ID PRIMARY KEY CONSTRAINT CHK_TBL_DEPARTMENTS_DEPT_ID CHECK(DEPT_ID LIKE 'D%'),
DEPT_NAME VARCHAR(100) UNIQUE NOT NULL
)
GO

INSERT INTO TBL_DEPARTMENTS VALUES ('D1001', 'INTERNAL MEDICINE')
INSERT INTO TBL_DEPARTMENTS VALUES ('D1002', 'OTOLOARYNGOLOGY')
INSERT INTO TBL_DEPARTMENTS VALUES ('D1003', 'GENERAL SURGERY')
INSERT INTO TBL_DEPARTMENTS VALUES ('D1004', 'PEDIATRICS')
INSERT INTO TBL_DEPARTMENTS VALUES ('D1005', 'CARDIOTHORACIC SURGERY')
INSERT INTO TBL_DEPARTMENTS VALUES ('D1006', 'PLASTIC SURGERY')
INSERT INTO TBL_DEPARTMENTS VALUES ('D1007', 'ORTHOPAEDICS')
GO

SELECT * FROM TBL_DEPARTMENTS
GO


--THIS TABLE STORES THE NAMES AND DEPARTMENTS OF THE DOCTORS
CREATE TABLE TBL_PHYSICIANS(
PHYSICIAN_ID INT IDENTITY (500, 1) CONSTRAINT PK_TBL_PHYSICIANS_PHYSICIAN_ID PRIMARY KEY,
PHYSICIAN_NAME VARCHAR(100) NOT NULL,
DEPT_ID VARCHAR(10) NOT NULL CONSTRAINT FK_TBL_PHYSICIANS_DEPARTMENT_ID REFERENCES TBL_DEPARTMENTS(DEPT_ID)
)
GO

INSERT INTO TBL_PHYSICIANS VALUES ('TOM WALTER', 'D1006')
INSERT INTO TBL_PHYSICIANS VALUES ('SAMANTHA JONES', 'D1006')
INSERT INTO TBL_PHYSICIANS VALUES ('NICHOLAS FLORRICK', 'D1006')
INSERT INTO TBL_PHYSICIANS VALUES ('CHRISTINA LOCKHART', 'D1006')
INSERT INTO TBL_PHYSICIANS VALUES ('SEAN HANKS', 'D1002')
INSERT INTO TBL_PHYSICIANS VALUES ('WALTER MCCLEAN', 'D1002')
INSERT INTO TBL_PHYSICIANS VALUES ('JACK SMITH', 'D1002')
INSERT INTO TBL_PHYSICIANS VALUES ('RYAN PARKER', 'D1001')
INSERT INTO TBL_PHYSICIANS VALUES ('SAMUEL GARDNER', 'D1001')
INSERT INTO TBL_PHYSICIANS VALUES ('WILL BLACK', 'D1001')
INSERT INTO TBL_PHYSICIANS VALUES ('MICHELLE CURTIS', 'D1001')
INSERT INTO TBL_PHYSICIANS VALUES ('ALICE GOLDBERG', 'D1001')
INSERT INTO TBL_PHYSICIANS VALUES ('DAWN HARDING', 'D1005')
INSERT INTO TBL_PHYSICIANS VALUES ('NATALIE DREW', 'D1005')
INSERT INTO TBL_PHYSICIANS VALUES ('BRIAN ADAMS', 'D1005')
INSERT INTO TBL_PHYSICIANS VALUES ('ANNA JOHNSON', 'D1004')
INSERT INTO TBL_PHYSICIANS VALUES ('DANIEL MASTERS', 'D1004')
INSERT INTO TBL_PHYSICIANS VALUES ('VICTORIA BAILEY', 'D1004')
INSERT INTO TBL_PHYSICIANS VALUES ('TIM SHAW', 'D1004')
INSERT INTO TBL_PHYSICIANS VALUES('ADAM SMITH', 'D1004')
INSERT INTO TBL_PHYSICIANS VALUES('ERIN JACOBS', 'D1006')
INSERT INTO TBL_PHYSICIANS VALUES('AMBER MCCARTHY', 'D1006')
INSERT INTO TBL_PHYSICIANS VALUES('CATHERINE BISHOP', 'D1006')
INSERT INTO TBL_PHYSICIANS VALUES('SUSAN MYERS', 'D1006')
INSERT INTO TBL_PHYSICIANS VALUES('JACKLYN BROWN', 'D1007')
INSERT INTO TBL_PHYSICIANS VALUES('GERRY SMITH', 'D1007')
INSERT INTO TBL_PHYSICIANS VALUES('DEAN JOHNSON', 'D1007')
INSERT INTO TBL_PHYSICIANS VALUES('JAY PETERSON', 'D1007')
INSERT INTO TBL_PHYSICIANS VALUES('ROSALINDA PARKS', 'D1003')
INSERT INTO TBL_PHYSICIANS VALUES('MARY FRANKLIN', 'D1003')
INSERT INTO TBL_PHYSICIANS VALUES('ROBERT CRAWLEY', 'D1003')
GO


SELECT * FROM TBL_PHYSICIANS
GO


--THIS TABLE STORES THE NAMES OF THE INSURANCE COMPANIES
CREATE TABLE TBL_INSURANCES
(
INSURANCE_ID INT IDENTITY(1000, 1) CONSTRAINT PK_TBL_INSURANCE_INSURANCE_ID PRIMARY KEY,
INSURANCE_NAME VARCHAR(50) UNIQUE NOT NULL,
FINANCIAL_CLASS VARCHAR(20) NOT NULL CONSTRAINT CHK_TBL_INSURANCE_FINANCIAL_CLASS CHECK (FINANCIAL_CLASS IN ('COMMERCIAL', 'MEDICARE', 'MEDICAID'))
)
GO

INSERT INTO TBL_INSURANCES VALUES ('AETNA', 'COMMERCIAL')
INSERT INTO TBL_INSURANCES VALUES ('BLUE CROSS BLUE SHIELD', 'COMMERCIAL')
INSERT INTO TBL_INSURANCES VALUES ('MEDICAID UTAH', 'MEDICAID')
INSERT INTO TBL_INSURANCES VALUES ('SELECT HEALTH', 'COMMERCIAL')
INSERT INTO TBL_INSURANCES VALUES ('SELECT HEALTH MEDICARE', 'MEDICARE')
INSERT INTO TBL_INSURANCES VALUES ('CIGNA', 'COMMERCIAL')
INSERT INTO TBL_INSURANCES VALUES ('MEDICARE', 'MEDICARE')
INSERT INTO TBL_INSURANCES VALUES ('SELECT HEALTH MEDICAID', 'MEDICAID')
INSERT INTO TBL_INSURANCES VALUES ('AETNA MEDICARE', 'MEDICARE')
INSERT INTO TBL_INSURANCES VALUES ('UNITED HEALTH CARE', 'COMMERCIAL')
GO

SELECT * FROM TBL_INSURANCES
GO


--THIS TABLE STORES THE DETAILS OF THE PATIENTS
CREATE TABLE TBL_PATIENTS(
PATIENT_ID INT IDENTITY(200, 1) CONSTRAINT PK_TBL_PATIENTS_PATIENT_ID PRIMARY KEY,
PATIENT_NAME VARCHAR(100) NOT NULL,
BIRTH_DATE DATE NOT NULL CONSTRAINT CHK_TBL_PATIENTS_BIRTH_DATE CHECK(BIRTH_DATE<=GETDATE()),
INSURANCE_ID INT NULL CONSTRAINT FK_TBL_TRANSACTIONS_INSURANCE_ID REFERENCES TBL_INSURANCES(INSURANCE_ID)
)
GO

INSERT INTO TBL_PATIENTS VALUES ('BENJAMIN LINCOLN', '09/01/1994', 1000)
INSERT INTO TBL_PATIENTS VALUES ('AARON GRAFF', '07/04/1976', 1001)
INSERT INTO TBL_PATIENTS VALUES ('STEPHEN GRANT', '04/15/1960', 1003)
INSERT INTO TBL_PATIENTS VALUES ('ANDREW ROGERS', '06/08/1982', 1002)
INSERT INTO TBL_PATIENTS VALUES ('KIM BOWERS', '09/02/2001', 1009)
INSERT INTO TBL_PATIENTS VALUES ('SARAH ARMSTRONG', '11/12/1978', NULL)
INSERT INTO TBL_PATIENTS VALUES ('CARL MAGNUS', '10/18/2012', 1007)
INSERT INTO TBL_PATIENTS VALUES ('TIMOTHY BRUNSWICK', '02/05/1992', 1002)
INSERT INTO TBL_PATIENTS VALUES ('PATRICK FULLER', '08/24/2015', 1005)
INSERT INTO TBL_PATIENTS VALUES ('CATHERINE LEWIS', '01/07/1965', NULL)
INSERT INTO TBL_PATIENTS VALUES ('PETER SUTTERBERG', '05/16/1961', 1004)
INSERT INTO TBL_PATIENTS VALUES ('MELISSA MANHEIM', '07/23/1975', 1008)
INSERT INTO TBL_PATIENTS VALUES ('CAROL STEWART', '03/18/1954', 1000)
INSERT INTO TBL_PATIENTS VALUES ('TODD SWEENEY', '05/12/2005', 1006)
INSERT INTO TBL_PATIENTS VALUES ('JACOB MUELLER', '09/05/2009', 1009)
INSERT INTO TBL_PATIENTS VALUES ('STACY NGUYEN', '12/20/1980', 1006)
INSERT INTO TBL_PATIENTS VALUES ('TRUMAN CAMPBELL', '06/15/2016', 1000)
INSERT INTO TBL_PATIENTS VALUES ('TAYLOR BARRYMORE', '04/12/1945', 1008)
INSERT INTO TBL_PATIENTS VALUES ('HENRIETTA BILLINGSLEY', '02/25/1949', 1003)
INSERT INTO TBL_PATIENTS VALUES ('JAMES PITT', '05/21/1969', 1002)
INSERT INTO TBL_PATIENTS VALUES ('KELSEY RICHARDS', '09/09/2009', 1007)
INSERT INTO TBL_PATIENTS VALUES ('MELISSA ANDREWS', '07/03/1995', 1004)
INSERT INTO TBL_PATIENTS VALUES ('CAROLINE CRAIG', '01/18/1953', 1005)
INSERT INTO TBL_PATIENTS VALUES ('CHARLOTTE GRAHAM', '05/11/2015', 1000)
INSERT INTO TBL_PATIENTS VALUES ('PAUL MUELLER', '09/05/2009', 1003)
INSERT INTO TBL_PATIENTS VALUES ('STEFANIE SMITH', '12/10/1981', NULL)
INSERT INTO TBL_PATIENTS VALUES ('WILLIAM HARRIS', '03/14/2006', 1001)
INSERT INTO TBL_PATIENTS VALUES ('LISA HOLDEN', '04/19/1985', 1005)
INSERT INTO TBL_PATIENTS VALUES ('HENRY HAWKING', '08/25/1989', 1003)
INSERT INTO TBL_PATIENTS VALUES ('JAMES FLEMING', '12/21/1998', 1001)
INSERT INTO TBL_PATIENTS VALUES ('RICHARD CASEY', '12/24/1969', 1001)
GO

SELECT * FROM TBL_PATIENTS
GO


--THIS TABLE STORES THE DETAILS OF ALL TRANSACTIONS ASSOCIATED WITH PATIENT VISITS
CREATE TABLE TBL_TRANSACTIONS(
TX_ID INT IDENTITY (1, 1) CONSTRAINT PK_TBL_TRANSACTIONS_TX_ID PRIMARY KEY,
VISIT_DATE DATE NOT NULL DEFAULT GETDATE() CONSTRAINT CHK_TBL_TRANSACTIONS_VISIT_DATE CHECK(VISIT_DATE<=GETDATE()),
PATIENT_ID INT NOT NULL CONSTRAINT FK_TBL_TRANSACTIONS_PATIENT_ID REFERENCES TBL_PATIENTS(PATIENT_ID),
PHYSICIAN_ID INT NOT NULL CONSTRAINT FK_TBL_TRANSACTIONS_PHYSICIAN_ID REFERENCES TBL_PHYSICIANS(PHYSICIAN_ID),
CHARGES MONEY NOT NULL CONSTRAINT CHK_TBL_TRANSACTIONS_CHARGES CHECK (CHARGES > 0),
INSURANCE_PAYMENT MONEY NULL,
PATIENT_PAYMENT MONEY NULL
)
GO

INSERT INTO TBL_TRANSACTIONS VALUES ('12/19/2016', 214, 515, 1830, 828, 692)
INSERT INTO TBL_TRANSACTIONS VALUES ('6/17/2018', 214, 516, 1787, 912, 238)
INSERT INTO TBL_TRANSACTIONS VALUES ('5/15/2016', 214, 516, 941, 639, 292)
INSERT INTO TBL_TRANSACTIONS VALUES ('3/12/2017', 214, 515, 968, 366, 602)
INSERT INTO TBL_TRANSACTIONS VALUES ('9/7/2017', 214, 515, 1954, 979, 433)
INSERT INTO TBL_TRANSACTIONS VALUES ('6/6/2017', 220, 518, 1562, 382, 921)
INSERT INTO TBL_TRANSACTIONS VALUES ('8/5/2017', 220, 518, 1924, 789, 334)
INSERT INTO TBL_TRANSACTIONS VALUES ('4/18/2016', 220, 518, 1766, 798, 882)
INSERT INTO TBL_TRANSACTIONS VALUES ('1/27/2016', 213, 515, 1154, 642, 425)
INSERT INTO TBL_TRANSACTIONS VALUES ('5/12/2016', 213, 524, 1683, 698, 903)
INSERT INTO TBL_TRANSACTIONS VALUES ('8/18/2017', 213, 524, 733, 225, 304)
INSERT INTO TBL_TRANSACTIONS VALUES ('3/7/2016', 206, 516, 1627, 308, 634)
INSERT INTO TBL_TRANSACTIONS VALUES ('7/11/2017', 206, 526, 1790, 727, 675)
INSERT INTO TBL_TRANSACTIONS VALUES ('2/27/2016', 208, 519, 1747, 762, 166)
INSERT INTO TBL_TRANSACTIONS VALUES ('9/13/2016', 208, 519, 1814, 285, 374)
INSERT INTO TBL_TRANSACTIONS VALUES ('8/28/2017', 208, 525, 1823, 592, 713)
INSERT INTO TBL_TRANSACTIONS VALUES ('11/3/2016', 224, 515, 1866, 157, 815)
INSERT INTO TBL_TRANSACTIONS VALUES ('12/24/2016', 224, 518, 1689, 26, 343)
INSERT INTO TBL_TRANSACTIONS VALUES ('3/23/2017', 224, 519, 1175, 446, 685)
INSERT INTO TBL_TRANSACTIONS VALUES ('10/11/2017', 224, 517, 1918, 316, 454)
INSERT INTO TBL_TRANSACTIONS VALUES ('1/28/2016', 226, 516, 1744, 154, 589)
INSERT INTO TBL_TRANSACTIONS VALUES ('5/22/2016', 226, 519, 1999, 932, 804)
INSERT INTO TBL_TRANSACTIONS VALUES ('2/22/2017', 226, 517, 1613, 32, 900)
INSERT INTO TBL_TRANSACTIONS VALUES ('8/15/2017', 226, 524, 1466, 213, 789)
INSERT INTO TBL_TRANSACTIONS VALUES ('10/5/2016', 216, 515, 1498, 757, 699)
INSERT INTO TBL_TRANSACTIONS VALUES ('4/6/2017', 216, 518, 1189, 810, 335)
INSERT INTO TBL_TRANSACTIONS VALUES ('5/31/2017', 223, 516, 1104, 480, 127)
INSERT INTO TBL_TRANSACTIONS VALUES ('11/16/2017', 223, 519, 758, 466, 212)
INSERT INTO TBL_TRANSACTIONS VALUES ('7/4/2016', 223, 517, 1949, 533, 639)
INSERT INTO TBL_TRANSACTIONS VALUES ('9/29/2017', 223, 517, 1412, 725, 439)
INSERT INTO TBL_TRANSACTIONS VALUES ('7/7/2017', 204, 516, 1567, 474, 241)
INSERT INTO TBL_TRANSACTIONS VALUES ('9/19/2017', 204, 519, 1999, 283, 863)
INSERT INTO TBL_TRANSACTIONS VALUES ('1/29/2016', 204, 517, 1753, 739, 645)
INSERT INTO TBL_TRANSACTIONS VALUES ('4/3/2017', 204, 524, 1075, 552, 392)
INSERT INTO TBL_TRANSACTIONS VALUES ('1/8/2016', 204, 515, 1134, 789, 250)
INSERT INTO TBL_TRANSACTIONS VALUES ('1/18/2017', 204, 515, 1167, 463, 647)
INSERT INTO TBL_TRANSACTIONS VALUES ('11/16/2016', 212, 500, 1954, 695, 834)
INSERT INTO TBL_TRANSACTIONS VALUES ('1/21/2017', 218, 524, 1661, 728, 466)
INSERT INTO TBL_TRANSACTIONS VALUES ('4/21/2016', 217, 504, 1090, 189, 657)
INSERT INTO TBL_TRANSACTIONS VALUES ('11/16/2017', 203, 529, 1324, 381, 884)
INSERT INTO TBL_TRANSACTIONS VALUES ('12/12/2016', 200, 512, 1644, 98, 901)
INSERT INTO TBL_TRANSACTIONS VALUES ('12/16/2016', 224, 512, 826, 135, 681)
INSERT INTO TBL_TRANSACTIONS VALUES ('8/30/2017', 226, 525, 1834, 788, 994)
INSERT INTO TBL_TRANSACTIONS VALUES ('1/21/2017', 225, 512, 1311, NULL, 365)
INSERT INTO TBL_TRANSACTIONS VALUES ('5/5/2016', 221, 504, 1147, 135, 944)
INSERT INTO TBL_TRANSACTIONS VALUES ('3/13/2017', 204, 524, 1859, 743, 578)
INSERT INTO TBL_TRANSACTIONS VALUES ('9/24/2017', 204, 530, 1080, 238, 116)
INSERT INTO TBL_TRANSACTIONS VALUES ('2/26/2016', 213, 508, 1906, 173, 105)
INSERT INTO TBL_TRANSACTIONS VALUES ('12/31/2016', 200, 528, 1005, 553, 168)
INSERT INTO TBL_TRANSACTIONS VALUES ('12/4/2016', 214, 510, 1911, 346, 873)
INSERT INTO TBL_TRANSACTIONS VALUES ('3/12/2017', 219, 528, 1615, 427, 220)
INSERT INTO TBL_TRANSACTIONS VALUES ('12/26/2016', 211, 510, 963, 308, 603)
INSERT INTO TBL_TRANSACTIONS VALUES ('5/24/2017', 215, 511, 1667, 117, 987)
INSERT INTO TBL_TRANSACTIONS VALUES ('6/13/2016', 223, 514, 1382, 29, 973)
INSERT INTO TBL_TRANSACTIONS VALUES ('5/25/2017', 206, 508, 1153, 10, 621)
INSERT INTO TBL_TRANSACTIONS VALUES ('2/3/2017', 205, 528, 1690, NULL, 634)
INSERT INTO TBL_TRANSACTIONS VALUES ('10/15/2017', 213, 503, 873, 22000, 625)
INSERT INTO TBL_TRANSACTIONS VALUES ('3/17/2016', 202, 500, 1640, 456, 820)
INSERT INTO TBL_TRANSACTIONS VALUES ('11/21/2017', 221, 521, 1900, 80035, 974)
INSERT INTO TBL_TRANSACTIONS VALUES ('2/1/2016', 201, 529, 518, 93, 200)
INSERT INTO TBL_TRANSACTIONS VALUES ('7/20/2016', 221, 508, 1771, 439, 794)
INSERT INTO TBL_TRANSACTIONS VALUES ('7/13/2017', 204, 512, 1974, 189, 355)
INSERT INTO TBL_TRANSACTIONS VALUES ('6/7/2016', 206, 500, 1798, 152, 504)
INSERT INTO TBL_TRANSACTIONS VALUES ('6/25/2016', 224, 522, 1840, 942, 376)
INSERT INTO TBL_TRANSACTIONS VALUES ('1/11/2016', 206, 506, 997, 169, 663)
INSERT INTO TBL_TRANSACTIONS VALUES ('5/29/2016', 207, 521, 1648, 788, 335)
INSERT INTO TBL_TRANSACTIONS VALUES ('1/6/2016', 208, 523, 1417, 361, 897)
INSERT INTO TBL_TRANSACTIONS VALUES ('10/8/2017', 205, 527, 1060, NULL, 474)
INSERT INTO TBL_TRANSACTIONS VALUES ('5/6/2017', 200, 512, 1166, 560, 182)
INSERT INTO TBL_TRANSACTIONS VALUES ('10/19/2016', 210, 530, 1872, 848, 962)
INSERT INTO TBL_TRANSACTIONS VALUES ('12/22/2017', 211, 524, 1831, 18310, 970)
INSERT INTO TBL_TRANSACTIONS VALUES ('7/15/2017', 207, 505, 1059, 733, 229)
INSERT INTO TBL_TRANSACTIONS VALUES ('12/29/2017', 215, 514, 1959, 10959, 674)
INSERT INTO TBL_TRANSACTIONS VALUES ('6/1/2017', 229, 530, 1942, 704, 586)
INSERT INTO TBL_TRANSACTIONS VALUES ('10/26/2016', 220, 501, 1928, 750, 302)
INSERT INTO TBL_TRANSACTIONS VALUES ('3/15/2016', 209, 525, 1656, NULL, 703)
INSERT INTO TBL_TRANSACTIONS VALUES ('9/21/2016', 229, 511, 848, 377, 331)
INSERT INTO TBL_TRANSACTIONS VALUES ('9/9/2016', 203, 508, 1886, 559, 151)
INSERT INTO TBL_TRANSACTIONS VALUES ('7/20/2016', 215, 504, 927, 112, 412)
INSERT INTO TBL_TRANSACTIONS VALUES ('6/21/2017', 224, 506, 1231, 262, 356)
INSERT INTO TBL_TRANSACTIONS VALUES ('4/30/2017', 204, 523, 1246, 107, 690)
INSERT INTO TBL_TRANSACTIONS VALUES ('5/8/2017', 201, 528, 1813, 303, 498)
INSERT INTO TBL_TRANSACTIONS VALUES ('9/12/2016', 213, 524, 1073, 898, 164)
INSERT INTO TBL_TRANSACTIONS VALUES ('6/15/2016', 222, 530, 1233, 15, 738)
INSERT INTO TBL_TRANSACTIONS VALUES ('9/25/2016', 213, 512, 1856, 830, 899)
INSERT INTO TBL_TRANSACTIONS VALUES ('6/17/2017', 211, 505, 1701, 497, 787)
INSERT INTO TBL_TRANSACTIONS VALUES ('10/13/2016', 217, 520, 1922, 965, 566)
INSERT INTO TBL_TRANSACTIONS VALUES ('12/15/2017', 218, 508, 1852, 778, 853)
INSERT INTO TBL_TRANSACTIONS VALUES ('11/5/2016', 228, 507, 1370, 609, 667)
INSERT INTO TBL_TRANSACTIONS VALUES ('4/10/2017', 221, 511, 1759, 877, 766)
INSERT INTO TBL_TRANSACTIONS VALUES ('6/27/2017', 218, 503, 1559, 805, 576)
INSERT INTO TBL_TRANSACTIONS VALUES ('2/19/2017', 228, 512, 1949, 717, 813)
INSERT INTO TBL_TRANSACTIONS VALUES ('11/5/2017', 227, 527, 1965, 680, 613)
INSERT INTO TBL_TRANSACTIONS VALUES ('3/19/2016', 203, 520, 1880, 491, 1389)
INSERT INTO TBL_TRANSACTIONS VALUES ('3/13/2017', 223, 504, 1109, 476, 377)
INSERT INTO TBL_TRANSACTIONS VALUES ('7/19/2017', 225, 525, 1785, NULL, 934)
INSERT INTO TBL_TRANSACTIONS VALUES ('6/16/2017', 216, 506, 1568, 812, 756)
INSERT INTO TBL_TRANSACTIONS VALUES ('9/16/2016', 201, 509, 1734, 874, 530)
INSERT INTO TBL_TRANSACTIONS VALUES ('7/13/2017', 230, 529, 964, 325, 191)
INSERT INTO TBL_TRANSACTIONS VALUES ('10/16/2017', 227, 507, 1883, 992, 832)
GO

SELECT * FROM TBL_TRANSACTIONS
GO

--1.
select p.PATIENT_ID,PATIENT_NAME, BIRTH_DATE, DEPT_NAME, count(distinct VISIT_DATE) as VISIT_COUNT
from TBL_PATIENTS p
inner join TBL_TRANSACTIONS t
on p.PATIENT_ID = t.PATIENT_ID 
inner join TBL_PHYSICIANS phys
on t.PHYSICIAN_ID = phys.PHYSICIAN_ID
inner join TBL_DEPARTMENTS
on phys.DEPT_ID = TBL_DEPARTMENTS.DEPT_ID
where (BIRTH_DATE <= ('1999-01-01')) and (VISIT_DATE between '2017-01-01' and '2017-12-31')
group by p.PATIENT_ID,DEPT_NAME,PATIENT_NAME,BIRTH_DATE, VISIT_DATE
order by VISIT_DATE asc;


--2. 
select d.DEPT_NAME,p.PATIENT_ID,PATIENT_NAME, INSURANCE_NAME, sum(INSURANCE_PAYMENT+PATIENT_PAYMENT)as outstanding_balance,COALESCE(i.INSURANCE_ID, 'SELF PAY') as INSURANCE_ID
from TBL_PATIENTS p
join TBL_TRANSACTIONS t
on p.PATIENT_ID = t.PATIENT_ID
join TBL_PHYSICIANS phys
on t.PHYSICIAN_ID = phys.PHYSICIAN_ID
join TBL_DEPARTMENTS d
on phys.DEPT_ID = d.DEPT_ID
join  TBL_INSURANCES i
on p.INSURANCE_ID = i.INSURANCE_ID
group by p.PATIENT_ID,d.DEPT_NAME, PATIENT_NAME, INSURANCE_NAME, i.INSURANCE_ID
order by outstanding_balance desc ;

--3. 
select PHYSICIAN_NAME, d.DEPT_NAME,sum(INSURANCE_PAYMENT) as total_insurance_payment, sum(INSURANCE_PAYMENT+PATIENT_PAYMENT)as outstanding_balance
from TBL_PATIENTS p
join TBL_TRANSACTIONS t
on p.PATIENT_ID = t.PATIENT_ID
join TBL_PHYSICIANS phys
on t.PHYSICIAN_ID = phys.PHYSICIAN_ID
join TBL_DEPARTMENTS d
on phys.DEPT_ID = d.DEPT_ID
join  TBL_INSURANCES i
on p.INSURANCE_ID = i.INSURANCE_ID
group by p.PATIENT_ID,d.DEPT_NAME, PHYSICIAN_NAME, INSURANCE_PAYMENT,phys.PHYSICIAN_ID
order by outstanding_balance desc, total_insurance_payment asc 
OFFSET 1 ROWS
FETCH NEXT 2 ROWS ONLY;


--4.
select TX_ID, TBL_TRANSACTIONS.VISIT_DATE, FINANCIAL_CLASS, sum(INSURANCE_PAYMENT + PATIENT_PAYMENT) as total_payment,COALESCE(TBL_INSURANCES.INSURANCE_ID, 'SELF PAY') as INSURANCE_ID
from TBL_INSURANCES, TBL_TRANSACTIONS
where (VISIT_DATE between '2016-01-01' and '2017-12-31')
group by TX_ID, FINANCIAL_CLASS, VISIT_DATE, INSURANCE_ID, DATEPART(MONTH, TBL_TRANSACTIONS.VISIT_DATE)
order by TBL_TRANSACTIONS.VISIT_DATE asc;


--5.Write a brief summary analyzing the information you see in the pivot table created in question 4. Identify any anomalies and find the TX_IDs that are causing these.
--Based on the pivot table, you can see insurance ids correlate with a specific financial class. Commerical Insurance brings n the most money due to the highest totaly payment sum. Some anomalies identified are that a couple of TX_IDs have a really low total payment compared to other ids. The TX_IDs causing these are TX_ID 48 and 60. TX_ID 76 also has a zero total payment for every category.

