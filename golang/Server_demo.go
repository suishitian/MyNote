package main

import (
	"fmt"
)

type A struct {
	Name string
}

func (a *A) setName(str string) {
	a.Name = str
}
func (a *A) getName() string {
	return a.Name
}

type B struct {
	A
	age int
}

func getB(str string, age int) *B {
	return &B{A{str}, age}
}
func (b *B) setAge(age int) {
	b.age = age
}
func (b *B) getAge() int {
	return b.age
}

//func (b *B) getName() string {
//	return B.Name
//}

//func (b *B) setName(str string) {
//	b.Info.setName(str)
//}
func main() {
	b := getB("heiheihie", 654654)
	b.setName("hahaha")
	b.setAge(123)
	fmt.Println(b.getName())
	fmt.Println(b.getAge())
}
