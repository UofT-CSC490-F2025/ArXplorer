variable "project_name" { type = string }
variable "vpc_cidr" { type = string }
variable "public_subnet_cidr" { type = string }
variable "public_subnet_cidr_b" { type = string }
variable "private_app_subnet_cidr" { type = string }
variable "private_ml_subnet_cidr" { type = string }
variable "private_data_subnet_cidr" { type = string }

data "aws_availability_zones" "available" {
  state = "available"
}

resource "aws_vpc" "main" {
  cidr_block           = var.vpc_cidr
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = { Name = "${var.project_name}-vpc" }
}

resource "aws_internet_gateway" "main" {
  vpc_id = aws_vpc.main.id
  tags   = { Name = "${var.project_name}-igw" }
}

resource "aws_subnet" "public" {
  vpc_id                  = aws_vpc.main.id
  cidr_block              = var.public_subnet_cidr
  availability_zone       = data.aws_availability_zones.available.names[0]
  map_public_ip_on_launch = true
  tags                    = { Name = "${var.project_name}-public-subnet" }
}

resource "aws_subnet" "public_b" {
  vpc_id                  = aws_vpc.main.id
  cidr_block              = var.public_subnet_cidr_b
  availability_zone       = data.aws_availability_zones.available.names[1]
  map_public_ip_on_launch = true
  tags                    = { Name = "${var.project_name}-public-subnet-b" }
}

resource "aws_subnet" "app_private" {
  vpc_id                  = aws_vpc.main.id
  cidr_block              = var.private_app_subnet_cidr
  availability_zone       = data.aws_availability_zones.available.names[0]
  map_public_ip_on_launch = false
  tags                    = { Name = "${var.project_name}-app-subnet" }
}

resource "aws_subnet" "ml_private" {
  vpc_id                  = aws_vpc.main.id
  cidr_block              = var.private_ml_subnet_cidr
  availability_zone       = data.aws_availability_zones.available.names[1]
  map_public_ip_on_launch = false
  tags                    = { Name = "${var.project_name}-ml-subnet" }
}

resource "aws_subnet" "data_private" {
  vpc_id                  = aws_vpc.main.id
  cidr_block              = var.private_data_subnet_cidr
  availability_zone       = data.aws_availability_zones.available.names[0]
  map_public_ip_on_launch = false
  tags                    = { Name = "${var.project_name}-data-subnet" }
}

resource "aws_route_table" "public" {
  vpc_id = aws_vpc.main.id
  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.main.id
  }
  tags = { Name = "${var.project_name}-public-rt" }
}

resource "aws_route_table_association" "public" {
  subnet_id      = aws_subnet.public.id
  route_table_id = aws_route_table.public.id
}

resource "aws_route_table_association" "public_b" {
  subnet_id      = aws_subnet.public_b.id
  route_table_id = aws_route_table.public.id
}

resource "aws_eip" "nat" {
  domain = "vpc"
  tags   = { Name = "${var.project_name}-nat-eip" }
}

resource "aws_nat_gateway" "main" {
  allocation_id = aws_eip.nat.id
  subnet_id     = aws_subnet.public.id
  depends_on    = [aws_internet_gateway.main]
  tags          = { Name = "${var.project_name}-nat" }
}

resource "aws_route_table" "private" {
  vpc_id = aws_vpc.main.id
  route {
    cidr_block     = "0.0.0.0/0"
    nat_gateway_id = aws_nat_gateway.main.id
  }
  tags = { Name = "${var.project_name}-private-rt" }
}

resource "aws_route_table_association" "app_private" {
  subnet_id      = aws_subnet.app_private.id
  route_table_id = aws_route_table.private.id
}

resource "aws_route_table_association" "ml_private" {
  subnet_id      = aws_subnet.ml_private.id
  route_table_id = aws_route_table.private.id
}

resource "aws_route_table_association" "data_private" {
  subnet_id      = aws_subnet.data_private.id
  route_table_id = aws_route_table.private.id
}

output "vpc_id" { value = aws_vpc.main.id }
output "public_subnet_ids" { value = [aws_subnet.public.id, aws_subnet.public_b.id] }
output "app_subnet_ids" { value = [aws_subnet.app_private.id] }
output "ml_subnet_ids" { value = [aws_subnet.ml_private.id] }
output "data_subnet_ids" { value = [aws_subnet.data_private.id] }
